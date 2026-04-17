"""Config loading utilities.

Training is configured entirely via YAML + CLI dot-notation overrides.
`load_config` merges the master training YAML with the referenced
corruption YAML (cfg.corruption.config_path) so the resulting DictConfig
has cfg.corruption populated with channel / severity / mode fields.
"""

from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(yaml_path: str, overrides: Optional[List[str]] = None) -> DictConfig:
    """Load YAML + CLI overrides + referenced corruption YAML.

    Merge order:
        1. Master YAML at yaml_path (expected structure: model / train /
           corruption / inference / wandb / ds_config).
        2. Corruption YAML at cfg.corruption.config_path, merged under
           cfg.corruption (preserves the `config_path` key for traceability).
        3. CLI dot-notation overrides (applied last; can override anything).

    Args:
        yaml_path: Path to the master training YAML.
        overrides: List of "key=value" strings, typically sys.argv leftovers.

    Returns:
        Merged DictConfig.
    """
    cfg = OmegaConf.load(yaml_path)

    corruption_path = cfg.get("corruption", {}).get("config_path", None)
    if corruption_path is not None:
        corruption_file = Path(corruption_path)
        if not corruption_file.is_absolute():
            # Resolve relative to repo root (parent of this file's parent).
            repo_root = Path(__file__).resolve().parent.parent
            corruption_file = repo_root / corruption_file
        if corruption_file.exists():
            corruption_cfg = OmegaConf.load(str(corruption_file))
            cfg.corruption = OmegaConf.merge(
                corruption_cfg,
                OmegaConf.create({"config_path": str(corruption_path)}),
            )

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
