from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    dataset_name: str = "RML2016.10a"
    data_path: str = "../data/RML2016.10a_dict.pkl"
    model_name: str = "resnet1d"
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    num_workers: int = 0
    save_dir: str = "results"
    random_seed: int = 42
    base_channels: int = 64
    dropout: float = 0.2
    early_stopping_patience: int = 5

    def save_path(self) -> Path:
        return Path(self.save_dir) / self.model_name
