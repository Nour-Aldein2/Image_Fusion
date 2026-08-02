## TODO: Test dilation as a way to improve the model, and compare your results to the results of the baseline
import torch
from rasterio.crs import defaultdict
from torch import nn
from tqdm import tqdm

from config import Config
from utils import CSMMask
from loss_fcns import CARLLoss
from data_utils import make_dataloaders, CloudRemovalDataset
from model import DSen2CR

cfg = Config()

criterion = CARLLoss(cfg)
csm_fcn = CSMMask(cfg)

csm = csm_fcn(cloudy_optical)
loss = criterion(pred, target, cloudy_input, csm)


def one_epoch(cfg: Config, model, optimizer,
              train_loader: CloudRemovalDataset, val_loader: CloudRemovalDataset,
              mask_calc: CSMMask, loss_fcn: CARLLoss):
    model.to(cfg.device)
    model.train()
    train_loss = 0.0
    train_samples = 0
    for b in tqdm(train_loader, desc="Training"):
        sar = b["sar"]
        cloudy = b["cloudy"]
        target = b["target"]
        sample_id = b["sample_id"]
        group = b["group"]
        roi = b["roi"]
        patch = b["patch"]
        mask = mask_calc(cloudy)

        optimizer.zero_grad()
        # Forward pass
        pred = model(sar, cloudy)
        # Calculate loss
        loss = loss_fcn(pred=pred, target=target, cloudy_input=cloudy, csm=mask)
        # Backprob
        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        train_loss += loss.item() * batch_size
        train_samples += batch_size

    train_loss /= train_samples

    model.eval()
    val_loss = 0.0
    val_samples = 0

    with torch.no_grad():
        for b in tqdm(val_loader, desc="Validation"):
            sar = b["sar"]
            cloudy = b["cloudy"]
            target = b["target"]
            sample_id = b["sample_id"]
            group = b["group"]
            roi = b["roi"]
            patch = b["patch"]
            mask = mask_calc(cloudy)

            # Forward pass
            pred = model(sar, cloudy)
            # Calculate loss
            loss = loss_fcn(pred=pred, target=target, cloudy_input=cloudy, csm=mask)

            batch_size = target.size(0)
            val_loss += loss.item() * batch_size
            val_samples += batch_size

        val_loss /= val_samples

    return train_loss, val_loss


if __name__ == "__main__":
    cfg = Config()
    model = DSen2CR(cfg)
    criterion = CARLLoss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    mask_calc = CSMMask(cfg)

    loaders, _, report = make_dataloaders(cfg.data_path,
                                          batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    torch.save(test_loader.dataset, "./test_dataset.pt")
    print(f"Saved test dataloader for later use to report and evaluate teh performance.")

    best_val_loss = float("inf")
    history = defaultdict(list)
    patience_counter = 0
    for epoch in range(cfg.epochs):
        train_loss, val_loss = one_epoch(cfg, model, optimizer, train_loader, val_loader, mask_calc, criterion)
        history["train"].append(train_loss.item())
        history["val"].append(val_loss.item())
        if val_loss < best_val_loss:
            print(f"New best model: val_loss={val_loss:.4f}, previous best={best_val_loss:.4f}")
            patience_counter = 0
            best_val_loss = val_loss
            torch.save( {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            }, f"best_model_epoch_{epoch}.pt")
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
            f"training_ended_epoch_{epoch}.pt")
            break

## TODO: when you get the dataloader, save the val and test splits for later use.
