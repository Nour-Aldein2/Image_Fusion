## TODO: Test dilation as a way to improve the model, and compare your results to the results of the baseline
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from utils import CSMMask
from loss_fcns import CARLLoss
from data_utils import make_dataloaders
from model import DSen2CR


## TODO: Double check how the initialised the parameters of the model and if it makes any difference
# def init_he_uniform(module: nn.Module) -> None:
#     """Initialize convolution weights as described in the paper."""
#     if isinstance(module, nn.Conv2d):
#         nn.init.kaiming_uniform_(
#             module.weight,
#             a=0.0,
#             mode="fan_in",
#             nonlinearity="relu",
#         )
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)

def one_epoch(cfg: Config, model, optimizer,
              train_loader: DataLoader, val_loader: DataLoader,
              mask_calc: CSMMask, loss_fcn: CARLLoss):
    device = torch.device(cfg.device)
    model.to(device)
    # -------------------------
    # Training
    # -------------------------
    model.train()
    train_loss = 0.0
    train_samples = 0
    for b in tqdm(train_loader, desc="Training"):
        sar = b["sar"].to(device, non_blocking=True)
        cloudy = b["cloudy"].to(device, non_blocking=True)
        target = b["target"].to(device, non_blocking=True)
        # CSMMask expects the unnormalized Sentinel-2 digital values.
        # It performs its own division by 10,000.
        cloudy_raw = b["cloudy_raw"].to(device, non_blocking=True)
        mask = mask_calc(cloudy_raw).to(device, non_blocking=True)

        optimizer.zero_grad()
        # Forward pass
        pred = model(sar, cloudy)
        # Calculate loss
        loss_output = loss_fcn(pred=pred, target=target, cloudy_input=cloudy, csm=mask)
        # Supports both return_parts=False and return_parts=True.
        loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
        # Backprob
        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        train_loss += loss.item() * batch_size
        train_samples += batch_size

    train_loss /= train_samples

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    val_loss = 0.0
    val_samples = 0

    with torch.no_grad():
        for b in tqdm(val_loader, desc="Validation"):
            sar = b["sar"].to(device, non_blocking=True)
            cloudy = b["cloudy"].to(device, non_blocking=True)
            target = b["target"].to(device, non_blocking=True)
            cloudy_raw = b["cloudy_raw"].to(device, non_blocking=True)
            mask = mask_calc(cloudy_raw).to(device, non_blocking=True)

            # Forward pass
            pred = model(sar, cloudy)
            # Calculate loss
            loss_output = loss_fcn(pred=pred, target=target, cloudy_input=cloudy, csm=mask)
            loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation loss for samples {b['sample_id']}")

            batch_size = target.size(0)
            val_loss += loss.item() * batch_size
            val_samples += batch_size

        val_loss /= val_samples

    return train_loss, val_loss


if __name__ == "__main__":
    cfg = Config()
    device = torch.device(cfg.device)
    model = DSen2CR(cfg).to(device)
    # model.apply(init_he_uniform)

    criterion = CARLLoss(cfg).to(device)
    mask_calc = CSMMask(cfg).to(device)
    mask_calc.eval()

    # "Adam with integrated Nesterov momentum" corresponds to NAdam.    # Paper's value
    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        momentum_decay=0.004,
    )

    loaders, datasets, report = make_dataloaders(
        cfg.data_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,

        # Sentinel-2: clip to [0, 10000], then divide by 2000.
        optical_scale=2000.0,
        optical_clip_max=10000.0,

        # Sentinel-1: clip using the existing VV/VH ranges and scale to [0, 2].
        sar_output_max=2.0,

        # The paper uses 149 train, 10 validation, and 10 test ROIs.
        train_fraction=149 / 169,
        val_fraction=10 / 169,
        test_fraction=10 / 169,
        split_unit="group_roi",

        seed=42,
        strict_discovery=True,
        augment_train=True,
        drop_last_train=True,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    print(f"Complete triplets found: {report.complete_triplets}")
    print(f"Training samples:   {len(datasets['train'])}")
    print(f"Validation samples: {len(datasets['val'])}")
    print(f"Test samples:       {len(datasets['test'])}")

    # Save stable identifiers instead of pickling the whole Dataset object.
    split_manifest = {
        split_name: [
            sample.sample_id
            for sample in dataset.samples
        ]
        for split_name, dataset in datasets.items()
    }
    torch.save(split_manifest, f"{cfg.saving_path}/split_manifest.pt")
    print("Saved train/validation/test sample IDs to split_manifest.pt")

    best_val_loss = float("inf")
    history = defaultdict(list)
    patience_counter = 0
    # ________ # Training Loop # _________ #
    for epoch in range(cfg.epochs):
        train_loss, val_loss = one_epoch(cfg, model, optimizer, train_loader, val_loader, mask_calc, criterion)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val_loss:
            print(f"New best model: val_loss={val_loss:.4f}, previous best={best_val_loss:.4f}")
            patience_counter = 0
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            }, f"{cfg.saving_path}/best_model_epoch_{epoch}.pt")
        else:
            patience_counter += 1

        if patience_counter >= cfg.es_patience:
            print(f" ------ Early stopping triggered at epoch {epoch} ------ ")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            },
                f"{cfg.saving_path}/training_ended_epoch_{epoch}.pt")
            break

## TODO: when you get the dataloader, save the val and test splits for later use.
