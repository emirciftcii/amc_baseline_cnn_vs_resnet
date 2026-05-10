from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import torch

from config import Config
from data.dataset import build_dataloaders_from_datasets, build_datasets, infer_data_spec, set_seed
from models import build_model
from training.engine import evaluate, fit, predict
from training.evaluate import (
    accuracy_by_snr,
    model_parameter_count,
    plot_history,
    plot_snr_accuracy,
    save_confusion_matrix,
)
from training.io import ensure_dir, save_checkpoint, save_json


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet1d", choices=["baseline_cnn", "resnet1d", "all"])
    parser.add_argument("--data-path", type=str, default="data/RML2016.10a_dict.pkl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    args = parser.parse_args()

    return Config(
        model_name=args.model,
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        early_stopping_patience=args.early_stopping_patience,
    )


def run_experiment(config: Config, model_name: str, device: torch.device) -> dict:
    config.model_name = model_name
    train_ds, val_ds, test_ds, le = build_datasets(config)
    spec = infer_data_spec(train_ds)
    print("Data spec:", spec, flush=True)
    print(f"Class count: {len(le)} | Classes: {le}", flush=True)

    train_loader, val_loader, test_loader, le = build_dataloaders_from_datasets(config, train_ds, val_ds, test_ds, le)

    num_classes = len(le)
    model = build_model(
        model_name=config.model_name,
        num_classes=num_classes,
        base_channels=config.base_channels,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    print(f"Training model={config.model_name} on device={device}", flush=True)
    start_time = time.perf_counter()
    model, history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=config.epochs,
        early_stopping_patience=config.early_stopping_patience,
    )
    training_time_sec = time.perf_counter() - start_time

    test_metrics = evaluate(model, test_loader, device)
    print("Test metrics:", test_metrics)
    y_true, y_pred, snrs = predict(model, test_loader, device)
    snr_scores = accuracy_by_snr(y_true, y_pred, snrs)

    save_root = ensure_dir(config.save_path())
    save_checkpoint(model, save_root / "best_model.pt")
    plot_history(history, save_root)
    plot_snr_accuracy(snr_scores, save_root)
    confusion = save_confusion_matrix(y_true, y_pred, list(le), save_root)
    save_json(history, save_root / "history.json")

    summary = {
        "config": asdict(config),
        "data_spec": spec,
        "classes": list(le),
        "best_epoch": history.get("best_epoch"),
        "best_val_loss": history.get("best_val_loss"),
        "test_metrics": test_metrics,
        "accuracy_by_snr": snr_scores,
        "confusion_matrix": confusion,
        "trainable_parameters": model_parameter_count(model),
        "training_time_sec": training_time_sec,
    }
    save_json(summary, save_root / "summary.json")
    return summary


def main() -> None:
    config = parse_args()
    set_seed(config.random_seed)

    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    model_names = ["baseline_cnn", "resnet1d"] if config.model_name == "all" else [config.model_name]
    summaries = {}

    for model_name in model_names:
        print(f"\n=== {model_name} ===", flush=True)
        set_seed(config.random_seed)
        summaries[model_name] = run_experiment(config, model_name, device)

    if len(summaries) > 1:
        comparison = {
            name: {
                "test_acc": summary["test_metrics"]["acc"],
                "test_loss": summary["test_metrics"]["loss"],
                "best_epoch": summary["best_epoch"],
                "trainable_parameters": summary["trainable_parameters"],
                "training_time_sec": summary["training_time_sec"],
            }
            for name, summary in summaries.items()
        }
        save_json(comparison, ensure_dir(config.save_dir) / "comparison_summary.json")
        print("Comparison:", comparison)


if __name__ == "__main__":
    main()
