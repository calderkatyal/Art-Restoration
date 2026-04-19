"""Dataset and dataloader utilities for art restoration training."""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .corruption import CorruptionModule

try:
    from torchdata.stateful_dataloader import StatefulDataLoader
except ImportError:  # pragma: no cover - optional dependency
    StatefulDataLoader = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TRAIN_SPLITS = {"train", "training"}
VAL_SPLITS = {"val", "valid", "validation", "eval", "test"}
LOGGER = logging.getLogger(__name__)


class ArtRestorationDataset(Dataset):
    """Training dataset for clean artwork plus synthetic corruption."""

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        corruption_config: Any,
        max_simultaneous: Optional[int] = None,
        split: str = "train",
        corruption_seed: int = 42,
        deterministic_corruption: Optional[bool] = None,
        return_metadata: bool = False,
    ):
        _validate_resolution(resolution)

        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir
        self.split = _canonicalize_split(split)
        self.resolution = int(resolution)
        self.return_metadata = bool(return_metadata)
        self.deterministic_corruption = (
            self.split != "train"
            if deterministic_corruption is None
            else bool(deterministic_corruption)
        )

        cfg = copy.deepcopy(corruption_config)
        if max_simultaneous == 1:
            cfg.individual_prob = 1.0
        self.corruptor = CorruptionModule(cfg)

        self.image_paths = _find_images(self.data_dir)
        self._base_corruption_seed = int(corruption_seed)
        self._corruption_rng = torch.Generator(device="cpu")
        self._worker_id = 0
        self._worker_seed = 0
        self._corruption_calls = 0
        self.configure_worker(worker_id=0)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_paths[idx]
        clean = self._load_and_preprocess(image_path)

        if self.deterministic_corruption:
            corruption_seed = _deterministic_seed_for_path(
                image_path=image_path,
                root=self.data_dir,
                index=idx,
                base_seed=self._base_corruption_seed,
            )
        else:
            corruption_seed = self._next_corruption_seed()

        corrupted, mask = self.corruptor(clean.clone(), seed=corruption_seed)
        sample: Dict[str, Any] = {
            "clean": clean,
            "corrupted": corrupted,
            "mask": mask,
        }

        if self.return_metadata:
            sample.update(
                {
                    "index": idx,
                    "path": str(image_path),
                    "genre": _genre_from_path(image_path, root=self.data_dir),
                    "corruption_seed": corruption_seed,
                }
            )

        return sample

    def configure_worker(self, worker_id: int) -> None:
        split_offset = 0 if self.split == "train" else 10_000_019
        self._worker_id = int(worker_id)
        self._worker_seed = self._base_corruption_seed + split_offset + worker_id * 1_009
        self._corruption_rng = torch.Generator(device="cpu")
        self._corruption_rng.manual_seed(self._worker_seed)
        self._corruption_calls = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self._worker_id,
            "worker_seed": self._worker_seed,
            "corruption_calls": self._corruption_calls,
            "corruption_rng_state": self._corruption_rng.get_state(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._worker_id = int(state_dict["worker_id"])
        self._worker_seed = int(state_dict["worker_seed"])
        self._corruption_calls = int(state_dict["corruption_calls"])
        self._corruption_rng = torch.Generator(device="cpu")
        self._corruption_rng.set_state(state_dict["corruption_rng_state"])

    def split_summary(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "image_dir": str(self.data_dir),
            "num_images": len(self.image_paths),
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

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        image = _open_image(path, mode="RGB")
        return _crop_resize_to_tensor(
            image,
            size=(self.resolution, self.resolution),
            interpolation=InterpolationMode.BICUBIC,
        )


class StatefulEpochSampler(Sampler[int]):
    """Single-process sampler that can resume mid-epoch."""

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

        order = _epoch_order(
            self._length,
            shuffle=self.shuffle,
            seed=self.seed,
            epoch=self.epoch,
        )
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


class DistributedStatefulEpochSampler(Sampler[int]):
    """DDP-aware sampler that keeps per-rank resume state."""

    def __init__(
        self,
        data_source: Sequence[Any],
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank must be in [0, num_replicas). Got rank={rank}, "
                f"num_replicas={num_replicas}."
            )

        self._length = len(data_source)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.num_samples = _distributed_num_samples(
            self._length,
            num_replicas=self.num_replicas,
            drop_last=self.drop_last,
        )
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0
        self.position = 0
        self._cached_epoch: Optional[int] = None
        self._cached_rank_order: Optional[List[int]] = None

    def __iter__(self) -> Iterator[int]:
        if self.position >= self.num_samples:
            self.advance_epoch()

        order = self.current_order()
        while self.position < self.num_samples:
            index = order[self.position]
            self.position += 1
            yield index

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "position": self.position,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "length": self._length,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "drop_last": self.drop_last,
            "num_samples": self.num_samples,
            "total_size": self.total_size,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        current_num_samples = _distributed_num_samples(
            self._length,
            num_replicas=self.num_replicas,
            drop_last=self.drop_last,
        )
        if int(state_dict["length"]) != self._length:
            raise ValueError(
                "Sampler length changed between save and resume. "
                f"Saved={state_dict['length']} current={self._length}."
            )
        if int(state_dict["num_replicas"]) != self.num_replicas:
            raise ValueError(
                "num_replicas changed between save and resume. "
                f"Saved={state_dict['num_replicas']} current={self.num_replicas}."
            )
        if int(state_dict["rank"]) != self.rank:
            raise ValueError(
                "rank changed between save and resume. "
                f"Saved={state_dict['rank']} current={self.rank}."
            )
        if bool(state_dict["drop_last"]) != self.drop_last:
            raise ValueError(
                "drop_last changed between save and resume. "
                f"Saved={state_dict['drop_last']} current={self.drop_last}."
            )
        if int(state_dict["num_samples"]) != current_num_samples:
            raise ValueError(
                "Distributed shard length changed between save and resume. "
                f"Saved={state_dict['num_samples']} current={current_num_samples}."
            )

        self.epoch = int(state_dict["epoch"])
        self.position = int(state_dict["position"])
        self.shuffle = bool(state_dict["shuffle"])
        self.seed = int(state_dict["seed"])
        self.num_samples = current_num_samples
        self.total_size = self.num_samples * self.num_replicas
        self._cached_epoch = None
        self._cached_rank_order = None

    def current_order(self) -> List[int]:
        if self._cached_rank_order is not None and self._cached_epoch == self.epoch:
            return self._cached_rank_order

        global_order = _epoch_order(
            self._length,
            shuffle=self.shuffle,
            seed=self.seed,
            epoch=self.epoch,
        )
        rank_order = _distribute_order(
            global_order,
            num_replicas=self.num_replicas,
            rank=self.rank,
            drop_last=self.drop_last,
        )
        self._cached_epoch = self.epoch
        self._cached_rank_order = rank_order
        return rank_order

    def seen_indices(self) -> List[int]:
        return self.current_order()[: self.position]

    def remaining_indices(self) -> List[int]:
        return self.current_order()[self.position :]

    def advance_epoch(self) -> None:
        self.epoch += 1
        self.position = 0
        self._cached_epoch = None
        self._cached_rank_order = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.position = 0
        self._cached_epoch = None
        self._cached_rank_order = None


def build_wikiart_dataloader(
    *,
    train_dir: str,
    val_dir: str,
    resolution: int,
    corruption_config: Any,
    batch_size: int,
    split: str = "train",
    max_simultaneous: Optional[int] = None,
    num_workers: int = 0,
    sampler_seed: int = 42,
    corruption_seed: int = 42,
    deterministic_corruption: Optional[bool] = None,
    shuffle: Optional[bool] = None,
    drop_last: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    snapshot_every_n_steps: int = 1,
    return_metadata: bool = False,
    distributed: bool = False,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
) -> tuple[ArtRestorationDataset, DataLoader, Sampler[int]]:
    """Build a resumable dataloader for either the train or val split."""
    canonical_split = _canonicalize_split(split)
    image_dir = train_dir if canonical_split == "train" else val_dir

    resolved_num_replicas: Optional[int] = None
    resolved_rank: Optional[int] = None
    if distributed:
        resolved_num_replicas, resolved_rank = _resolve_distributed_context(
            num_replicas=num_replicas,
            rank=rank,
        )

    dataset = ArtRestorationDataset(
        data_dir=image_dir,
        resolution=resolution,
        corruption_config=corruption_config,
        max_simultaneous=max_simultaneous,
        split=canonical_split,
        corruption_seed=_offset_seed_for_rank(corruption_seed, resolved_rank),
        deterministic_corruption=deterministic_corruption,
        return_metadata=return_metadata,
    )

    sampler_shuffle = (canonical_split == "train") if shuffle is None else bool(shuffle)
    sampler_base_seed = int(sampler_seed) + (0 if canonical_split == "train" else 1_000_003)

    if distributed:
        sampler: Sampler[int] = DistributedStatefulEpochSampler(
            dataset,
            num_replicas=int(resolved_num_replicas),
            rank=int(resolved_rank),
            shuffle=sampler_shuffle,
            seed=sampler_base_seed,
            drop_last=drop_last,
        )
    else:
        sampler = StatefulEpochSampler(
            dataset,
            shuffle=sampler_shuffle,
            seed=sampler_base_seed,
        )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "sampler": sampler,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
        "persistent_workers": bool(persistent_workers) and int(num_workers) > 0,
        "worker_init_fn": _wikiart_worker_init_fn if int(num_workers) > 0 else None,
    }
    if StatefulDataLoader is not None:
        dataloader = StatefulDataLoader(
            snapshot_every_n_steps=int(snapshot_every_n_steps),
            **loader_kwargs,
        )
    else:
        dataloader = DataLoader(**loader_kwargs)

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
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.resolution = resolution
        self.num_mask_channels = num_mask_channels
        self.image_paths = _find_images(self.data_dir)
        self.mask_paths = _build_stem_index(self.mask_dir) if self.mask_dir is not None else {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.image_paths[idx]
        damaged = self._load_and_preprocess(path)
        h, w = damaged.shape[-2:]

        if self.mask_dir is None:
            mask = torch.ones((self.num_mask_channels, h, w), dtype=torch.float32)
        else:
            mask_path = self.mask_paths.get(path.stem)
            if mask_path is None:
                LOGGER.warning(
                    "No mask found for image '%s' in '%s'; using all-ones mask.",
                    path.name,
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
        f"Unknown split '{split}'. Expected one of {sorted(TRAIN_SPLITS | VAL_SPLITS)}."
    )


def _find_images(root: Path) -> List[Path]:
    if not root.exists():
        raise ValueError(f"Image root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Image root is not a directory: {root}")

    image_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    )
    if not image_paths:
        raise ValueError(f"No images found under: {root}")
    return image_paths


def _genre_from_path(path: Path, *, root: Path) -> str:
    relative_parts = path.relative_to(root).parts
    return relative_parts[0] if len(relative_parts) > 1 else "__root__"


def _deterministic_seed_for_path(
    *,
    image_path: Path,
    root: Path,
    index: int,
    base_seed: int,
) -> int:
    path_hash = _stable_string_hash(str(image_path.relative_to(root)))
    return int((int(base_seed) + index * 1_009 + path_hash) % (2**31 - 1))


def _stable_string_hash(value: str) -> int:
    total = 0
    for char in value:
        total = (total * 257 + ord(char)) % (2**31 - 1)
    return total


def _validate_resolution(resolution: int) -> None:
    if int(resolution) <= 0:
        raise ValueError("resolution must be > 0")
    if int(resolution) % 16 != 0:
        raise ValueError("resolution must be divisible by 16")


def _open_image(path: Path, mode: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert(mode)
    except Exception as exc:  # pragma: no cover - filesystem dependent
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
    return TF.to_tensor(image).float().clamp(0.0, 1.0)


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
                f"Multiple mask files found for stem '{stem}': {index[stem]} and {path}"
            )
        index[stem] = path

    return index


def _epoch_order(length: int, *, shuffle: bool, seed: int, epoch: int) -> List[int]:
    if not shuffle:
        return list(range(length))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(epoch))
    return torch.randperm(length, generator=generator).tolist()


def _distributed_num_samples(length: int, *, num_replicas: int, drop_last: bool) -> int:
    if drop_last and length % num_replicas != 0:
        return math.ceil((length - num_replicas) / num_replicas)
    return math.ceil(length / num_replicas)


def _distribute_order(
    order: Sequence[int],
    *,
    num_replicas: int,
    rank: int,
    drop_last: bool,
) -> List[int]:
    if drop_last:
        total_size = _distributed_num_samples(
            len(order),
            num_replicas=num_replicas,
            drop_last=True,
        ) * num_replicas
        distributed = list(order[:total_size])
    else:
        num_samples = _distributed_num_samples(
            len(order),
            num_replicas=num_replicas,
            drop_last=False,
        )
        total_size = num_samples * num_replicas
        distributed = list(order)
        if total_size > len(distributed):
            padding = total_size - len(distributed)
            if padding <= len(distributed):
                distributed.extend(distributed[:padding])
            else:
                repeats = math.ceil(padding / len(distributed))
                distributed.extend((distributed * repeats)[:padding])

    return distributed[rank:total_size:num_replicas]


def _resolve_distributed_context(
    *,
    num_replicas: Optional[int],
    rank: Optional[int],
) -> tuple[int, int]:
    if num_replicas is None or rank is None:
        if not dist.is_available() or not dist.is_initialized():
            raise ValueError(
                "Distributed dataloader requested without explicit "
                "`num_replicas`/`rank`, and torch.distributed is not initialized."
            )
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

    return int(num_replicas), int(rank)


def _offset_seed_for_rank(seed: int, rank: Optional[int]) -> int:
    if rank is None:
        return int(seed)
    return int(seed) + int(rank) * 1_000_003
