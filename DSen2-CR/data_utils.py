"""PyTorch data loader for paired Sentinel-1 / Sentinel-2 cloud-removal data.

Expected structure (one or more seasons/groups):

    data/
      ROIs1158_spring_s1/
        s1_58/ROIs1158_spring_s1_58_p100.tif
      ROIs1158_spring_s2/
        s2_58/ROIs1158_spring_s2_58_p100.tif
      ROIs1158_spring_s2_cloudy/
        s2_cloudy_58/ROIs1158_spring_s2_cloudy_58_p100.tif

The loader pairs files by (group, ROI id, patch id), not by directory order.
It returns normalized tensors for DSen2CR/CARL and the raw cloudy optical
image separately for CSMMask.
"""

from __future__ import annotations

import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# Match top-level roots such as ROIs1158_spring_s1 and
# ROIs1158_spring_s2_cloudy. Order matters: s2_cloudy must precede s2.
_ROOT_RE = re.compile(
    r"^(?P<group>ROIs\d+_.+?)_(?P<modality>s2_cloudy|s2|s1)$",
    flags=re.IGNORECASE,
)

# Match the common tail in all three modalities, e.g. _58_p100.tif.
_PATCH_RE = re.compile(
    r"_(?P<roi>\d+)_p(?P<patch>\d+)\.tiff?$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Triplet:
    """Paths belonging to one aligned training example."""

    group: str
    roi: int
    patch: int
    sar_path: Path
    target_path: Path
    cloudy_path: Path

    @property
    def sample_id(self) -> str:
        return f"{self.group}_roi{self.roi}_p{self.patch}"


@dataclass(frozen=True)
class DiscoveryReport:
    complete_triplets: int
    groups_seen: Tuple[str, ...]
    missing_roots: Mapping[str, Tuple[str, ...]]
    unmatched_files: Mapping[str, Mapping[str, int]]


def _index_tiffs(root: Path) -> Dict[Tuple[int, int], Path]:
    """Index TIFFs beneath one modality root by (ROI, patch)."""
    indexed: Dict[Tuple[int, int], Path] = {}

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue

        match = _PATCH_RE.search(path.name)
        if match is None:
            warnings.warn(f"Ignoring TIFF with unrecognised name: {path}")
            continue

        key = (int(match.group("roi")), int(match.group("patch")))
        if key in indexed:
            raise ValueError(
                f"Duplicate ROI/patch key {key} under {root}:\n"
                f"  {indexed[key]}\n"
                f"  {path}"
            )
        indexed[key] = path

    return indexed


def discover_triplets(
    data_root: str | Path,
    *,
    strict: bool = False,
) -> Tuple[list[Triplet], DiscoveryReport]:
    """Discover aligned SAR, clean-S2, and cloudy-S2 files.

    Parameters
    ----------
    data_root:
        Directory containing roots such as ``ROIs1158_spring_s1``.
    strict:
        If True, fail when a group is missing a modality root or when any
        modality has an unmatched file. If False, use only the intersection
        of the three modalities and report the omissions.
    """
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root does not exist: {data_root}")

    roots: Dict[str, Dict[str, Path]] = {}
    for child in data_root.iterdir():
        if not child.is_dir():
            continue
        match = _ROOT_RE.match(child.name)
        if match is None:
            continue
        group = match.group("group")
        modality = match.group("modality").lower()
        roots.setdefault(group, {})[modality] = child

    if not roots:
        raise RuntimeError(
            f"No modality roots found under {data_root}. Expected names like "
            "ROIs1158_spring_s1, ROIs1158_spring_s2, and "
            "ROIs1158_spring_s2_cloudy."
        )

    required = {"s1", "s2", "s2_cloudy"}
    missing_roots: Dict[str, Tuple[str, ...]] = {}
    unmatched_files: Dict[str, Dict[str, int]] = {}
    samples: list[Triplet] = []

    for group in sorted(roots):
        group_roots = roots[group]
        missing = tuple(sorted(required - set(group_roots)))
        if missing:
            missing_roots[group] = missing
            message = f"Skipping {group}; missing roots: {', '.join(missing)}"
            if strict:
                raise RuntimeError(message)
            warnings.warn(message)
            continue

        sar = _index_tiffs(group_roots["s1"])
        target = _index_tiffs(group_roots["s2"])
        cloudy = _index_tiffs(group_roots["s2_cloudy"])

        common = set(sar) & set(target) & set(cloudy)
        counts = {
            "s1_only_or_unmatched": len(set(sar) - common),
            "s2_only_or_unmatched": len(set(target) - common),
            "s2_cloudy_only_or_unmatched": len(set(cloudy) - common),
        }
        unmatched_files[group] = counts

        if strict and any(counts.values()):
            raise RuntimeError(
                f"Unmatched TIFFs in {group}: {counts}. "
                "Finish extraction or call discover_triplets(..., strict=False)."
            )

        for roi, patch in sorted(common):
            samples.append(
                Triplet(
                    group=group,
                    roi=roi,
                    patch=patch,
                    sar_path=sar[(roi, patch)],
                    target_path=target[(roi, patch)],
                    cloudy_path=cloudy[(roi, patch)],
                )
            )

    if not samples:
        raise RuntimeError(
            "No complete SAR/clean/cloudy triplets were found. Check extraction "
            "status and file naming."
        )

    report = DiscoveryReport(
        complete_triplets=len(samples),
        groups_seen=tuple(sorted(roots)),
        missing_roots=missing_roots,
        unmatched_files=unmatched_files,
    )
    return samples, report


def split_triplets(
    samples: Sequence[Triplet],
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    split_unit: Literal["roi", "group_roi", "sample"] = "roi",
) -> Dict[str, list[Triplet]]:
    """Deterministically split examples without patch-level leakage by default.

    ``split_unit='roi'`` keeps every patch from the same numeric ROI in one
    split, including that ROI across multiple seasons. This is the safest
    default for spatial generalisation.
    """
    fractions = (train_fraction, val_fraction, test_fraction)
    if any(x < 0 for x in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Split fractions must be non-negative and sum to 1; got {fractions}.")

    def unit(sample: Triplet):
        if split_unit == "roi":
            return sample.roi
        if split_unit == "group_roi":
            return sample.group, sample.roi
        if split_unit == "sample":
            return sample.sample_id
        raise ValueError(f"Unsupported split_unit: {split_unit}")

    units = sorted({unit(sample) for sample in samples}, key=str)
    rng = random.Random(seed)
    rng.shuffle(units)

    n_units = len(units)
    n_train = int(n_units * train_fraction)
    n_val = int(n_units * val_fraction)

    train_units = set(units[:n_train])
    val_units = set(units[n_train : n_train + n_val])
    test_units = set(units[n_train + n_val :])

    result = {"train": [], "val": [], "test": []}
    for sample in samples:
        sample_unit = unit(sample)
        if sample_unit in train_units:
            result["train"].append(sample)
        elif sample_unit in val_units:
            result["val"].append(sample)
        elif sample_unit in test_units:
            result["test"].append(sample)
        else:  # pragma: no cover - defensive check
            raise AssertionError(f"Unassigned split unit: {sample_unit}")

    return result


def _read_tiff(path: Path, expected_bands: int) -> Tensor:
    """Read a TIFF as contiguous float32 [C,H,W] and repair non-finite values."""
    with rasterio.open(path, "r") as src:
        array = src.read(out_dtype="float32")

    if array.ndim != 3 or array.shape[0] != expected_bands:
        raise ValueError(
            f"Expected {expected_bands} bands in {path}, got shape {array.shape}."
        )

    if not np.isfinite(array).all():
        array = array.copy()
        for channel in range(array.shape[0]):
            band = array[channel]
            finite = np.isfinite(band)
            fill = float(band[finite].mean()) if finite.any() else 0.0
            band[~finite] = fill

    return torch.from_numpy(np.ascontiguousarray(array))


class CloudRemovalDataset(Dataset):
    """
    Load aligned Sentinel-1 and Sentinel-2 image triplets.

    Each dataset item contains a two-band Sentinel-1 SAR image, a
    thirteen-band cloudy Sentinel-2 image, and the corresponding clean
    Sentinel-2 target.

    Cropping and augmentation are applied identically to all three
    images to preserve their spatial alignment.

    The SAR and optical images are normalized independently. The raw
    cloudy optical image can also be returned for cloud-mask generation.

    Notes
    -----
    The normalized ``cloudy`` tensor should be passed to ``DSen2CR`` and
    ``CARLLoss``. The unnormalized ``cloudy_raw`` tensor should be passed
    to ``CSMMask``.
    """

    def __init__(
        self,
        samples: Sequence[Triplet],
        *,
        crop_size: Optional[int | Tuple[int, int]] = None,
        random_crop: bool = False,
        augment: bool = False,
        optical_scale: float = 2000.0,
        optical_clip_min: float = 0.0,
        optical_clip_max: Optional[float] = None,
        sar_clip: Sequence[Tuple[float, float]] = ((-25.0, 0.0), (-32.5, 0.0)),
        sar_output_max: float = 5.0,
        return_cloudy_raw: bool = True,
    ) -> None:
        """Initialize the cloud-removal dataset.

        Parameters
        ----------
        samples:
            Aligned SAR, clean optical, and cloudy optical triplets.
        crop_size:
            Output crop size as an integer or ``(height, width)``.
            ``None`` keeps the complete image.
        random_crop:
            Select a random crop when ``True``. Otherwise, use a centered
            crop.
        augment:
            Apply synchronized random flips and rotations.
        optical_scale:
            Value used to scale the optical tensors.
        optical_clip_min:
            Minimum value retained in the optical tensors.
        optical_clip_max:
            Maximum value retained in the optical tensors. ``None``
            disables upper clipping.
        sar_clip:
            Clipping ranges for the two SAR channels.
        sar_output_max:
            Maximum value of each normalized SAR channel.
        return_cloudy_raw:
            Include the unnormalized cloudy optical tensor in each item.

        Raises
        ------
        ValueError
            If ``samples`` is empty, ``optical_scale`` is not positive,
            or ``sar_clip`` does not contain two ranges.
        """
        if not samples:
            raise ValueError("CloudRemovalDataset received an empty sample list.")
        if optical_scale <= 0:
            raise ValueError("optical_scale must be positive.")
        if len(sar_clip) != 2:
            raise ValueError("sar_clip must contain exactly two (min,max) pairs.")

        self.samples = list(samples)
        self.crop_size = self._normalise_crop_size(crop_size)
        self.random_crop = random_crop
        self.augment = augment
        self.optical_scale = float(optical_scale)
        self.optical_clip_min = float(optical_clip_min)
        self.optical_clip_max = optical_clip_max
        self.sar_clip = tuple((float(lo), float(hi)) for lo, hi in sar_clip)
        self.sar_output_max = float(sar_output_max)
        self.return_cloudy_raw = return_cloudy_raw

    @staticmethod
    def _normalise_crop_size(
        crop_size: Optional[int | Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """Validate and standardize a crop-size specification.

        Parameters
        ----------
        crop_size:
            Crop size as an integer, ``(height, width)``, or ``None``.

        Returns
        -------
        tuple of int or None
            A ``(height, width)`` tuple, or ``None`` when cropping is
            disabled.

        Raises
        ------
        ValueError
            If the crop size is not two-dimensional or contains a
            non-positive value.
        """
        if crop_size is None:
            return None
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        if len(crop_size) != 2 or min(crop_size) <= 0:
            raise ValueError(f"Invalid crop_size: {crop_size}")
        return int(crop_size[0]), int(crop_size[1])

    def __len__(self) -> int:
        return len(self.samples)

    def _crop(self, tensors: Iterable[Tensor]) -> list[Tensor]:
        tensors = list(tensors)
        if self.crop_size is None:
            return tensors

        height, width = tensors[0].shape[-2:]
        crop_h, crop_w = self.crop_size
        if crop_h > height or crop_w > width:
            raise ValueError(
                f"Crop {self.crop_size} exceeds image size {(height, width)}."
            )

        if self.random_crop:
            top = int(torch.randint(0, height - crop_h + 1, (1,)).item())
            left = int(torch.randint(0, width - crop_w + 1, (1,)).item())
        else:
            top = (height - crop_h) // 2
            left = (width - crop_w) // 2

        return [
            tensor[..., top : top + crop_h, left : left + crop_w]
            for tensor in tensors
        ]

    def _augment(self, tensors: Iterable[Tensor]) -> list[Tensor]:
        tensors = list(tensors)
        if not self.augment:
            return tensors

        if torch.rand(()) < 0.5:
            tensors = [torch.flip(tensor, dims=(-1,)) for tensor in tensors]
        if torch.rand(()) < 0.5:
            tensors = [torch.flip(tensor, dims=(-2,)) for tensor in tensors]

        rotations = int(torch.randint(0, 4, (1,)).item())
        if rotations:
            tensors = [torch.rot90(tensor, rotations, dims=(-2, -1)) for tensor in tensors]

        return [tensor.contiguous() for tensor in tensors]

    def _normalise_sar(self, sar: Tensor) -> Tensor:
        """Clip and normalize a two-channel SAR tensor.

        Each SAR channel is clipped to its configured range and mapped
        linearly to ``[0, sar_output_max]``.

        Parameters
        ----------
        sar:
            SAR tensor with shape ``[2, H, W]``.

        Returns
        -------
        Tensor
            Normalized SAR tensor with the same shape.

        Raises
        ------
        ValueError
            If a SAR clipping range has an upper bound less than or equal
            to its lower bound.
        """
        sar = sar.clone()
        for channel, (lower, upper) in enumerate(self.sar_clip):
            if upper <= lower:
                raise ValueError(f"Invalid SAR clipping range: {(lower, upper)}")
            sar[channel].clamp_(lower, upper)
            sar[channel].sub_(lower).div_(upper - lower).mul_(self.sar_output_max)
        return sar

    def _normalise_optical(self, optical: Tensor) -> Tensor:
        """Clip and scale a Sentinel-2 optical tensor.

        Parameters
        ----------
        optical:
            Optical tensor with shape ``[13, H, W]``.

        Returns
        -------
        Tensor
            Clipped tensor divided by ``self.optical_scale``.
        """
        optical = optical.clone()
        if self.optical_clip_max is None:
            optical.clamp_(min=self.optical_clip_min)
        else:
            optical.clamp_(min=self.optical_clip_min, max=float(self.optical_clip_max))
        return optical / self.optical_scale

    def __getitem__(self, index: int):
        """Load and prepare one aligned image triplet.

        Parameters
        ----------
        index:
            Position of the sample in the dataset.

        Returns
        -------
        dict
            Dictionary containing:

            ``sar``
                Normalized SAR tensor with shape ``[2, H, W]``.
            ``cloudy``
                Normalized cloudy optical tensor with shape
                ``[13, H, W]``.
            ``target``
                Normalized clean optical tensor with shape
                ``[13, H, W]``.
            ``cloudy_raw``
                Unnormalized cloudy optical tensor. This key is omitted
                when ``return_cloudy_raw`` is ``False``.
            ``sample_id``
                Unique sample identifier.
            ``group``
                Dataset group or seasonal subset.
            ``roi``
                Region-of-interest identifier.
            ``patch``
                Patch identifier.

        Raises
        ------
        ValueError
            If the SAR, cloudy, and target images have different spatial
            dimensions.
        """
        sample = self.samples[index]

        sar = _read_tiff(sample.sar_path, expected_bands=2)
        target_raw = _read_tiff(sample.target_path, expected_bands=13)
        cloudy_raw = _read_tiff(sample.cloudy_path, expected_bands=13)

        if sar.shape[-2:] != target_raw.shape[-2:] or sar.shape[-2:] != cloudy_raw.shape[-2:]:
            raise ValueError(
                f"Spatial mismatch for {sample.sample_id}: "
                f"sar={tuple(sar.shape)}, target={tuple(target_raw.shape)}, "
                f"cloudy={tuple(cloudy_raw.shape)}"
            )

        sar, target_raw, cloudy_raw = self._crop((sar, target_raw, cloudy_raw))
        sar, target_raw, cloudy_raw = self._augment((sar, target_raw, cloudy_raw))

        result = {
            "sar": self._normalise_sar(sar),
            "cloudy": self._normalise_optical(cloudy_raw),
            "target": self._normalise_optical(target_raw),
            "sample_id": sample.sample_id,
            "group": sample.group,
            "roi": sample.roi,
            "patch": sample.patch,
        }
        if self.return_cloudy_raw:
            result["cloudy_raw"] = cloudy_raw.float()

        return result


def _seed_worker(worker_id: int) -> None:
    # DataLoader gives each worker a unique torch seed. Reuse it for Python/NumPy.
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_dataloaders(
    data_root: str | Path,
    *,
    batch_size: int,
    num_workers: int = 4,
    crop_size: Optional[int | Tuple[int, int]] = None,
    optical_scale: float = 2000.0,
    optical_clip_max: Optional[float] = None,
    sar_output_max: float = 5.0,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    split_unit: Literal["roi", "group_roi", "sample"] = "roi",
    seed: int = 42,
    strict_discovery: bool = False,
    augment_train: bool = True,
    drop_last_train: bool = True,
    pin_memory: Optional[bool] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, CloudRemovalDataset], DiscoveryReport]:
    """Discover samples and create training, validation, and test loaders.

    Parameters
    ----------
    data_root:
        Root directory containing the seasonal SAR, clean optical, and
        cloudy optical folders.
    batch_size:
        Number of samples returned in each batch.
    num_workers:
        Number of subprocesses used for data loading.
    crop_size:
        Output crop size. ``None`` keeps the complete image.
    optical_scale:
        Value used to scale Sentinel-2 tensors.
    optical_clip_max:
        Maximum retained optical value. ``None`` disables upper clipping.
    sar_output_max:
        Maximum value of each normalized SAR channel.
    train_fraction:
        Fraction of splitting units assigned to training.
    val_fraction:
        Fraction of splitting units assigned to validation.
    test_fraction:
        Fraction of splitting units assigned to testing.
    split_unit:
        Unit used to prevent leakage between subsets.

        ``"roi"``
            Keep each ROI number in one subset.
        ``"group_roi"``
            Keep each group and ROI combination in one subset.
        ``"sample"``
            Split individual image patches independently.
    seed:
        Random seed used for splitting and DataLoader shuffling.
    strict_discovery:
        Raise an error when incomplete image triplets are found.
    augment_train:
        Apply synchronized augmentation to the training dataset.
    drop_last_train:
        Drop the final incomplete training batch.
    pin_memory:
        Enable pinned CPU memory. When ``None``, enable it automatically
        if CUDA is available.

    Returns
    -------
    loaders:
        Dictionary containing ``train``, ``val``, and ``test``
        DataLoaders.
    datasets:
        Dictionary containing the corresponding dataset objects.
    report:
        Discovery report describing complete and incomplete triplets.

    Raises
    ------
    ValueError
        If the split fractions are invalid, a subset is empty, or a
        dataset configuration is invalid.
    FileNotFoundError
        If the data root or required dataset folders do not exist.

    Notes
    -----
    ``split_unit="group_roi"`` is recommended for the seasonal dataset.
    It prevents patches from the same seasonal ROI from appearing in
    multiple subsets.
    """
    samples, report = discover_triplets(data_root, strict=strict_discovery)
    splits = split_triplets(
        samples,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        split_unit=split_unit,
    )

    datasets = {
        "train": CloudRemovalDataset(
            splits["train"],
            crop_size=crop_size,
            random_crop=crop_size is not None,
            augment=augment_train,
            optical_scale=optical_scale,
            optical_clip_max=optical_clip_max,
            sar_output_max=sar_output_max,
        ),
        "val": CloudRemovalDataset(
            splits["val"],
            crop_size=crop_size,
            random_crop=False,
            augment=False,
            optical_scale=optical_scale,
            optical_clip_max=optical_clip_max,
            sar_output_max=sar_output_max,
        ),
        "test": CloudRemovalDataset(
            splits["test"],
            crop_size=crop_size,
            random_crop=False,
            augment=False,
            optical_scale=optical_scale,
            optical_clip_max=optical_clip_max,
            sar_output_max=sar_output_max,
        ),
    }

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        "persistent_workers": num_workers > 0,
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            drop_last=drop_last_train,
            generator=generator,
            **common_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        ),
    }

    return loaders, datasets, report