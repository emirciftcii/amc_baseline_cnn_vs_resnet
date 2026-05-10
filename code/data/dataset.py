from __future__ import annotations

import pickle
import random
import warnings
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import Config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_data_path(data_path: str) -> Path:
    p = Path(data_path)
    if p.exists():
        return p.resolve()
    code_dir = Path(__file__).resolve().parents[1]
    candidate = (code_dir / data_path).resolve()
    if candidate.exists():
        return candidate
    data_dir_candidate = (code_dir / "data" / p.name).resolve()
    if data_dir_candidate.exists():
        return data_dir_candidate
    raise FileNotFoundError(
        f"Dataset file not found. Checked: {p}, {candidate}, and {data_dir_candidate}. "
        "Download RML2016.10a_dict.pkl manually and pass --data-path."
    )


def load_radioml_2016a_local(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    path = _resolve_data_path(data_path)
    print(f"Loading dataset from: {path}", flush=True)
    with open(path, "rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*VisibleDeprecationWarning.*")
            warnings.filterwarnings("ignore", message=".*align should be passed.*")
            data = pickle.load(f, encoding="latin1")

    if not isinstance(data, dict) or not data:
        raise ValueError("Expected the RadioML 2016.10A pickle to contain a non-empty dict.")

    bad_keys = [key for key in data.keys() if not isinstance(key, tuple) or len(key) != 2]
    if bad_keys:
        raise ValueError(f"Unexpected RadioML key format. Example bad key: {bad_keys[0]!r}")

    mods = sorted({key[0] for key in data.keys()})
    snrs = sorted({int(key[1]) for key in data.keys()})
    mod_to_idx = {m: i for i, m in enumerate(mods)}

    X_list, y_list, snr_list = [], [], []
    for mod in mods:
        for snr in snrs:
            if (mod, snr) not in data:
                continue
            block = np.asarray(data[(mod, snr)], dtype=np.float32)
            if block.ndim != 3 or 2 not in block.shape[1:]:
                raise ValueError(
                    f"Expected each block to look like [N, 2, 128] or [N, 128, 2], got {block.shape} for {(mod, snr)}"
                )
            X_list.append(block)
            y_list.append(np.full(len(block), mod_to_idx[mod], dtype=np.int64))
            snr_list.append(np.full(len(block), snr, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    snr = np.concatenate(snr_list)
    print(f"Loaded {len(X)} samples across {len(mods)} classes and {len(snrs)} SNR values.", flush=True)
    return X, y, snr, mods


def _make_tensor_dataset(X: np.ndarray, y: np.ndarray, snr: np.ndarray, indices: Sequence[int]) -> TensorDataset:
    idx = np.asarray(indices)
    return TensorDataset(
        torch.tensor(X[idx], dtype=torch.float32),
        torch.tensor(y[idx], dtype=torch.long),
        torch.tensor(snr[idx], dtype=torch.long),
    )


def build_datasets(config: Config):
    X, y, snr, le = load_radioml_2016a_local(config.data_path)

    print("Creating train/validation/test splits...", flush=True)
    idx = np.arange(len(X))
    stratify_key = np.array([f"{label}_{snr_value}" for label, snr_value in zip(y, snr)])
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, stratify_key, test_size=0.30, random_state=config.random_seed, stratify=stratify_key
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.50, random_state=config.random_seed, stratify=y_temp
    )

    train = _make_tensor_dataset(X, y, snr, train_idx)
    val = _make_tensor_dataset(X, y, snr, val_idx)
    test = _make_tensor_dataset(X, y, snr, test_idx)
    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}", flush=True)
    return train, val, test, le


def build_dataloaders(config: Config):
    train, val, test, le = build_datasets(config)
    return build_dataloaders_from_datasets(config, train, val, test, le)


def build_dataloaders_from_datasets(config: Config, train, val, test, le):

    pin_memory = torch.cuda.is_available() and config.device == "cuda"
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader, le


def infer_data_spec(dataset) -> dict:
    x, y, snr = dataset[0]
    return {
        "x_shape": tuple(x.shape),
        "y_shape": tuple(torch.as_tensor(y).shape),
        "snr_shape": tuple(torch.as_tensor(snr).shape),
        "n_classes_guess": None,
    }
