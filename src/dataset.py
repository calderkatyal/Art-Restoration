"""Dataset and dataloader utilities for art restoration training.

`ArtRestorationDataset` loads clean WikiArt images, applies synthetic
corruption on-the-fly, and exposes deterministic 80/20 split handling from a
single root directory. The split is stratified by the top-level genre folder.

`StatefulEpochSampler` is designed to work with
`torchdata.stateful_dataloader.StatefulDataLoader` so training can resume
mid-epoch and still know exactly which examples have already been consumed.

`RealDamageDataset` remains available for future real-damage evaluation, but
no dedicated dataloader helper is implemented for it yet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler, get_worker_info
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import CorruptionConfig
from .corruption import CorruptionModule

try:
    from torchdata.stateful_dataloader import StatefulDataLoader
except ImportError:  # pragma: no cover - optional dependency in this repo
    StatefulDataLoader = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TRAIN_SPLITS = {"train", "training"}
VAL_SPLITS = {"val", "valid", "validation", "eval", "test"}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageRecord:
    """One WikiArt image and its top-level genre label."""

    path: Path
    genre: str


@dataclass(frozen=True)
class WikiArtSplitInfo:
    """Metadata describing the deterministic WikiArt train/eval split."""

    split: str
    total_images: int
    num_train: int
    num_eval: int
    train_ratio: float
    split_seed: int


class ArtRestorationDataset(Dataset):
    """WikiArt training/evaluation dataset with synthetic corruption.

    Each item returns:
        clean:          (3, H, W) float32 in [0, 1]
        corrupted:      (3, H, W) float32 in [0, 1]
        mask:           (K, H, W) float32 in [0, 1]
        index:          integer dataset index within this split
        path:           absolute image path as string
        genre:          top-level genre folder name
        corruption_seed integer seed used for this item's corruption call
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        corruption_config: CorruptionConfig,
        split: Literal["train", "val", "eval", "test"] = "train",
        train_ratio: float = 0.8,
        split_seed: int = 42,
        corruption_seed: int = 42,
        deterministic_corruption: Optional[bool] = None,
        return_metadata: bool = True,
    ):
        _validate_resolution(resolution)
        _validate_split_ratio(train_ratio)

        canonical_split = _canonicalize_split(split)
        root = Path(data_dir)
        all_records = _find_wikiart_records(root)
        train_records, eval_records = _split_wikiart_records(
            all_records,
            train_ratio=train_ratio,
            seed=split_seed,
        )

        selected_records = train_records if canonical_split == "train" else eval_records
        if not selected_records:
            raise ValueError(
                f"No images available for split '{canonical_split}' under '{data_dir}'."
            )

        self.data_dir = root
        self.resolution = resolution
        self.split = canonical_split
        self.train_ratio = train_ratio
        self.split_seed = int(split_seed)
        self.deterministic_corruption = (
            canonical_split != "train"
            if deterministic_corruption is None
            else bool(deterministic_corruption)
        )
        self.return_metadata = return_metadata
        self.records = selected_records
        self.image_paths = [record.path for record in self.records]
        self.corruption = CorruptionModule(corruption_config)
        self.split_info = WikiArtSplitInfo(
            split=self.split,
            total_images=len(all_records),
            num_train=len(train_records),
            num_eval=len(eval_records),
            train_ratio=train_ratio,
            split_seed=int(split_seed),
        )

        self._base_corruption_seed = int(corruption_seed)
        self._corruption_rng = torch.Generator(device="cpu")
        self._worker_id = 0
        self._worker_seed = 0
        self._corruption_calls = 0
        self.configure_worker(worker_id=0)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        clean = self._load_and_preprocess(record.path)
        corruption_seed = (
            self._deterministic_corruption_seed(record, idx)
            if self.deterministic_corruption
            else self._next_corruption_seed()
        )
        corrupted, mask = self.corruption(clean.clone(), seed=corruption_seed)

        sample: Dict[str, Any] = {
            "clean": clean,
            "corrupted": corrupted,
            "mask": mask,
        }
        if self.return_metadata:
            sample.update(
                {
                    "index": idx,
                    "path": str(record.path),
                    "genre": record.genre,
                    "corruption_seed": corruption_seed,
                }
            )
        return sample

    def configure_worker(self, worker_id: int) -> None:
        """Initialize deterministic corruption RNG state for one worker."""
        split_offset = 0 if self.split == "train" else 10_000_019
        self._worker_id = int(worker_id)
        self._worker_seed = self._base_corruption_seed + split_offset + worker_id * 1_009
        self._corruption_rng.manual_seed(self._worker_seed)
        self._corruption_calls = 0

    def state_dict(self) -> Dict[str, Any]:
        """Persist dataset RNG state for exact mid-epoch corruption resumption."""
        return {
            "worker_id": self._worker_id,
            "worker_seed": self._worker_seed,
            "corruption_calls": self._corruption_calls,
            "corruption_rng_state": self._corruption_rng.get_state(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore dataset RNG state."""
        self._worker_id = int(state_dict["worker_id"])
        self._worker_seed = int(state_dict["worker_seed"])
        self._corruption_calls = int(state_dict["corruption_calls"])
        self._corruption_rng = torch.Generator(device="cpu")
        self._corruption_rng.set_state(state_dict["corruption_rng_state"])

    def split_summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of the selected split."""
        return {
            "split": self.split_info.split,
            "total_images": self.split_info.total_images,
            "num_train": self.split_info.num_train,
            "num_eval": self.split_info.num_eval,
            "train_ratio": self.split_info.train_ratio,
            "split_seed": self.split_info.split_seed,
        }

    def _next_corruption_seed(self) -> int:
        self._corruption_calls += 1
        return int(
            torch.randint(
                low=0,
                high=2**31,
                size=(1,),
                generator=self._corruption_rng,
            ).item()
        )

    def _deterministic_corruption_seed(self, record: ImageRecord, idx: int) -> int:
        path_hash = _stable_string_hash(str(record.path.relative_to(self.data_dir)))
        return int(
            (
                self._base_corruption_seed
                + self.split_seed * 10_007
                + idx * 1_009
                + path_hash
            )
            % (2**31 - 1)
        )

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        image = _open_image(path, mode="RGB")
        return _crop_resize_to_tensor(
            image,
            size=(self.resolution, self.resolution),
            interpolation=InterpolationMode.BICUBIC,
        )


class StatefulEpochSampler(Sampler[int]):
    """Deterministic sampler with resumable intra-epoch position tracking.

    The sampler tracks:
        epoch:      current epoch number
        position:   number of samples already yielded in the current epoch
        shuffle:    whether the epoch order is shuffled

    The current epoch order is deterministic from `seed + epoch`, so the state
    only needs the epoch/position pair. This is enough both to resume training
    mid-epoch and to recover the exact examples already seen during that epoch.
    """

    def __init__(
        self,
        data_source: Sequence[Any],
        *,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self._length = len(data_source)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.position = 0
        self._cached_epoch: Optional[int] = None
        self._cached_order: Optional[List[int]] = None

    def __iter__(self) -> Iterator[int]:
        if self.position >= self._length:
            self.advance_epoch()

        order = self.current_order()
        while self.position < self._length:
            index = order[self.position]
            self.position += 1
            yield index

    def __len__(self) -> int:
        return self._length

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "position": self.position,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "length": self._length,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if int(state_dict["length"]) != self._length:
            raise ValueError(
                "Sampler length changed between save and resume. "
                f"Saved={state_dict['length']} current={self._length}."
            )
        self.epoch = int(state_dict["epoch"])
        self.position = int(state_dict["position"])
        self.shuffle = bool(state_dict["shuffle"])
        self.seed = int(state_dict["seed"])
        self._cached_epoch = None
        self._cached_order = None

    def current_order(self) -> List[int]:
        if self._cached_order is not None and self._cached_epoch == self.epoch:
            return self._cached_order

        if not self.shuffle:
            order = list(range(self._length))
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(self._length, generator=generator).tolist()

        self._cached_epoch = self.epoch
        self._cached_order = order
        return order

    def seen_indices(self) -> List[int]:
        return self.current_order()[: self.position]

    def remaining_indices(self) -> List[int]:
        return self.current_order()[self.position :]

    def advance_epoch(self) -> None:
        self.epoch += 1
        self.position = 0
        self._cached_epoch = None
        self._cached_order = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.position = 0
        self._cached_epoch = None
        self._cached_order = None


def build_wikiart_dataloader(
    *,
    data_dir: str,
    resolution: int,
    corruption_config: CorruptionConfig,
    split: Literal["train", "val", "eval", "test"] = "train",
    batch_size: int,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    split_seed: int = 42,
    corruption_seed: int = 42,
    deterministic_corruption: Optional[bool] = None,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    snapshot_every_n_steps: int = 1,
    return_metadata: bool = True,
):
    """Build a WikiArt dataset plus a torchdata StatefulDataLoader.

    Returns:
        (dataset, dataloader, sampler)

    Raises:
        ImportError if `torchdata` is not installed.
    """
    if StatefulDataLoader is None:
        raise ImportError(
            "torchdata is required for build_wikiart_dataloader(). "
            "Install it with `pip install torchdata`."
        )

    canonical_split = _canonicalize_split(split)
    dataset = ArtRestorationDataset(
        data_dir=data_dir,
        resolution=resolution,
        corruption_config=corruption_config,
        split=canonical_split,
        train_ratio=train_ratio,
        split_seed=split_seed,
        corruption_seed=corruption_seed,
        deterministic_corruption=deterministic_corruption,
        return_metadata=return_metadata,
    )
    sampler = StatefulEpochSampler(
        dataset,
        shuffle=(canonical_split == "train") if shuffle is None else shuffle,
        seed=split_seed + (0 if canonical_split == "train" else 1_000_003),
    )

    dataloader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(canonical_split == "train") if drop_last is None else drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
        worker_init_fn=_wikiart_worker_init_fn if num_workers > 0 else None,
        snapshot_every_n_steps=snapshot_every_n_steps,
    )
    return dataset, dataloader, sampler


class RealDamageDataset(Dataset):
    """Evaluation dataset for real damaged artwork (e.g. MuralDH)."""

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        mask_dir: Optional[str] = None,
        num_mask_channels: int = 8,
    ):
        _validate_resolution(resolution)
        if num_mask_channels < 1:
            raise ValueError("num_mask_channels must be >= 1")

        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.num_mask_channels = num_mask_channels
        self.image_paths = _find_images(data_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.mask_paths = (
            _build_stem_index(self.mask_dir) if self.mask_dir is not None else {}
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        damaged = self._load_and_preprocess(image_path)
        h, w = damaged.shape[-2:]

        if self.mask_dir is None:
            mask = torch.ones((self.num_mask_channels, h, w), dtype=torch.float32)
        else:
            mask_path = self.mask_paths.get(image_path.stem)
            if mask_path is None:
                LOGGER.warning(
                    "No mask found for image '%s' in '%s'; using all-ones mask.",
                    image_path.name,
                    self.mask_dir,
                )
                mask = torch.ones((self.num_mask_channels, h, w), dtype=torch.float32)
            else:
                mask = self._load_mask(mask_path, h=h, w=w)

        return {"damaged": damaged, "mask": mask}

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        image = _open_image(path, mode="RGB")
        return _crop_resize_to_tensor(
            image,
            size=(self.resolution, self.resolution),
            interpolation=InterpolationMode.BICUBIC,
        )

    def _load_mask(self, path: Path, h: int, w: int) -> torch.Tensor:
        mask_image = _open_image(path, mode="L")
        mask = _crop_resize_to_tensor(
            mask_image,
            size=(h, w),
            interpolation=InterpolationMode.NEAREST,
        )
        mask = (mask > 0).to(dtype=torch.float32)
        return mask.repeat(self.num_mask_channels, 1, 1)


def _wikiart_worker_init_fn(worker_id: int) -> None:
    """Seed each worker's dataset-local corruption RNG."""
    worker_info = get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    if hasattr(dataset, "configure_worker"):
        dataset.configure_worker(worker_id)


def _canonicalize_split(split: str) -> str:
    split_lower = split.lower()
    if split_lower in TRAIN_SPLITS:
        return "train"
    if split_lower in VAL_SPLITS:
        return "eval"
    raise ValueError(
        f"Unknown split '{split}'. Expected one of "
        f"{sorted(TRAIN_SPLITS | VAL_SPLITS)}."
    )


def _find_wikiart_records(root: Path) -> List[ImageRecord]:
    """Recursively find WikiArt images and annotate them by top-level genre."""
    if not root.exists():
        raise ValueError(f"Image root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Image root is not a directory: {root}")

    records: List[ImageRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        relative_parts = path.relative_to(root).parts
        genre = relative_parts[0] if len(relative_parts) > 1 else "__root__"
        records.append(ImageRecord(path=path, genre=genre))

    if not records:
        raise ValueError(f"No images found under: {root}")
    return records


def _split_wikiart_records(
    records: Sequence[ImageRecord],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[List[ImageRecord], List[ImageRecord]]:
    """Stratified split by top-level genre folder."""
    by_genre: Dict[str, List[ImageRecord]] = {}
    for record in records:
        by_genre.setdefault(record.genre, []).append(record)

    train_records: List[ImageRecord] = []
    eval_records: List[ImageRecord] = []

    for genre in sorted(by_genre):
        genre_records = by_genre[genre]
        shuffled = _shuffle_records(genre_records, seed=seed + _stable_string_hash(genre))
        n_total = len(shuffled)
        if n_total == 1:
            n_train = 1
        else:
            n_train = int(n_total * train_ratio)
            n_train = min(max(n_train, 1), n_total - 1)

        train_records.extend(shuffled[:n_train])
        eval_records.extend(shuffled[n_train:])

    return sorted(train_records, key=lambda record: record.path), sorted(
        eval_records, key=lambda record: record.path
    )


def _shuffle_records(records: Sequence[ImageRecord], seed: int) -> List[ImageRecord]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(len(records), generator=generator).tolist()
    return [records[i] for i in order]


def _stable_string_hash(value: str) -> int:
    total = 0
    for char in value:
        total = (total * 257 + ord(char)) % (2**31 - 1)
    return total


def _find_images(root: str) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise ValueError(f"Image root does not exist: {root}")
    if not root_path.is_dir():
        raise ValueError(f"Image root is not a directory: {root}")

    image_paths = sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    )
    if not image_paths:
        raise ValueError(f"No images found under: {root}")
    return image_paths


def _validate_split_ratio(train_ratio: float) -> None:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")


def _validate_resolution(resolution: int) -> None:
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    if resolution % 16 != 0:
        raise ValueError("resolution must be divisible by 16")


def _open_image(path: Path, mode: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert(mode)
    except Exception as exc:  # pragma: no cover - depends on filesystem assets
        raise RuntimeError(f"Failed to load image: {path}") from exc


def _crop_resize_to_tensor(
    image: Image.Image,
    size: tuple[int, int],
    interpolation: InterpolationMode,
) -> torch.Tensor:
    width, height = image.size
    crop_size = min(width, height)
    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)

    image = TF.crop(image, top=top, left=left, height=crop_size, width=crop_size)
    image = TF.resize(
        image,
        size=list(size),
        interpolation=interpolation,
        antialias=interpolation != InterpolationMode.NEAREST,
    )
    return TF.to_tensor(image)


def _build_stem_index(root: Path) -> Dict[str, Path]:
    if not root.exists():
        raise ValueError(f"Mask root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Mask root is not a directory: {root}")

    index: Dict[str, Path] = {}
    mask_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    )
    if not mask_paths:
        LOGGER.warning("No mask files found under '%s'; using all-ones fallback.", root)
        return index

    for path in mask_paths:
        stem = path.stem
        if stem in index:
            raise ValueError(
                f"Multiple mask files found for stem '{stem}': "
                f"{index[stem]} and {path}"
            )
        index[stem] = path

    return index
