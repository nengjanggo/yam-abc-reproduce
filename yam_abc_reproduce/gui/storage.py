"""Per-GUI storage defaults, independent of backend working directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StoragePaths:
    root: Path = REPO_ROOT

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    def resolve(self, path: str | Path) -> Path:
        """Relative output overrides are relative to the selected save root."""
        return (self.root / Path(path).expanduser()).resolve()

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def model(self) -> Path:
        return self.root / "model"

    @property
    def episodes(self) -> Path:
        return self.data / "episodes"

    @property
    def rollouts(self) -> Path:
        return self.data / "rollouts"

    @property
    def lerobot(self) -> Path:
        return self.data / "lerobot"

    @property
    def abc_cache(self) -> Path:
        return self.data / "abc_cache"

    @property
    def openpi_assets(self) -> Path:
        return self.data / "openpi_assets"

    def checkpoints(self, backend: str) -> Path:
        family = "openpi" if backend in ("pi0", "pi05") else backend
        return self.model / family / "checkpoints"

    def model_cache(self, backend: str) -> Path:
        return self.model / backend / "cache"

    def cache_env(self) -> dict[str, str]:
        # Set storage variables, not HF_HOME: credentials stay in their existing location.
        hub = str(self.model / "huggingface" / "hub")
        return {
            "HF_LEROBOT_HOME": str(self.lerobot),
            "HF_DATASETS_CACHE": str(self.data / "huggingface" / "datasets"),
            "HF_HUB_CACHE": hub,
            "HUGGINGFACE_HUB_CACHE": hub,
            "TRANSFORMERS_CACHE": hub,
            "HF_XET_CACHE": str(self.model / "huggingface" / "xet"),
            "HF_ASSETS_CACHE": str(self.model / "huggingface" / "assets"),
            "HF_MODULES_CACHE": str(self.model / "huggingface" / "modules"),
            "OPENPI_DATA_HOME": str(self.model_cache("openpi")),
            "MOLMOACT2_CHECKPOINT_CACHE": str(self.model_cache("molmoact2") / "archives"),
            "CACHED_PATH_CACHE_ROOT": str(self.model_cache("molmoact2") / "downloads"),
            "TORCH_HOME": str(self.model / "torch"),
        }

    def conversion_params(self, params: dict, episodes: Path | None = None) -> dict:
        """Freeze the exact input/output locations for both launch and progress."""
        p = dict(params)
        task = (p.get("task") or p.get("dataset") or "").strip()
        src = p.get("src") or ((episodes or self.episodes) / task if task else None)
        if src is None:
            raise ValueError(f"convert needs a task (folder under {episodes or self.episodes}) or an explicit src path")
        fmt = p.get("to", "lerobot")
        if fmt not in ("lerobot", "abc"):
            raise ValueError(f"unknown conversion format {fmt!r}")
        repo_id = (p.get("repo_id") or task or "yam_abc_reproduce/dataset").strip()
        p.update(src=str(self.resolve(src)), repo_id=repo_id,
                 out=str(self.resolve(p.get("out") or self.data / fmt / repo_id)))
        return p


DEFAULT_PATHS = StoragePaths()
