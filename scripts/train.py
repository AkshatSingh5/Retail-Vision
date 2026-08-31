"""Train YOLO26m on the Retail Vision custom product dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import ROOT_DIR as PROJECT_ROOT
from scripts.dataset_common import detect_device

DEFAULT_MODEL = PROJECT_ROOT / "vision" / "models" / "yolo26m.pt"
DEFAULT_DATA = PROJECT_ROOT / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs"


def next_experiment_name(project: Path, prefix: str = "retail_yolo26m") -> str:
    project.mkdir(parents=True, exist_ok=True)
    version = 1
    while (project / f"{prefix}_v{version}").exists():
        version += 1
    return f"{prefix}_v{version}"


def update_env_model_path(relative_path: str) -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith("MODEL_PATH="):
            new_lines.append(f"MODEL_PATH={relative_path}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"MODEL_PATH={relative_path}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26m on custom retail products.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Pretrained YOLO26m weights")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--device", default=None, help="cuda device id like 0, or cpu. Auto-detected if omitted.")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers. Auto if omitted.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Output directory root")
    parser.add_argument("--name", default=None, help="Experiment name. Auto-increments retail_yolo26m_vN if omitted.")
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mosaic", type=float, default=0.8)
    parser.add_argument("--mixup", type=float, default=0.05)
    parser.add_argument("--freeze", type=int, default=0, help="Freeze first N layers; 0 = none")
    parser.add_argument("--lr0", type=float, default=None, help="Initial LR. Auto if omitted.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing run folder")
    return parser.parse_args()


def train(args: argparse.Namespace) -> Path:
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    os.chdir(PROJECT_ROOT)

    device = detect_device(args.device)
    if args.workers is None:
        workers = 0 if sys.platform.startswith("win") else min(8, torch.get_num_threads())
    else:
        workers = args.workers

    name = args.name or next_experiment_name(args.project)
    run_dir = args.project / name
    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"Run already exists: {run_dir}. Pass --name or omit it to auto-increment.")

    config = {
        "model": str(args.model),
        "data": str(args.data),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "epochs": args.epochs,
        "device": device,
        "workers": workers,
        "project": str(args.project),
        "name": name,
        "patience": args.patience,
        "seed": args.seed,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "augmentation": {
            "degrees": 18.0,
            "translate": 0.15,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0005,
            "fliplr": 0.5,
            "flipud": 0.05,
            "hsv_h": 0.015,
            "hsv_s": 0.6,
            "hsv_v": 0.5,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "erasing": 0.25,
            "close_mosaic": 10,
            "freeze": args.freeze,
        },
    }

    print(f"Device: {device}")
    print(f"Experiment: {name}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        device=device,
        workers=workers,
        project=str(args.project),
        name=name,
        exist_ok=args.exist_ok,
        patience=args.patience,
        seed=args.seed,
        pretrained=True,
        plots=True,
        val=True,
        cache=True,
        cos_lr=True,
        optimizer="auto",
        degrees=18.0,
        translate=0.15,
        scale=0.5,
        shear=0.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.05,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.5,
        mosaic=args.mosaic,
        mixup=args.mixup,
        erasing=0.25,
        close_mosaic=10 if args.mosaic > 0 else 0,
        freeze=args.freeze or None,
        verbose=True,
    )

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    exported = PROJECT_ROOT / "vision" / "models" / f"{name}.pt"
    if best.exists():
        exported.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, exported)
        update_env_model_path(f"vision/models/{name}.pt")
        config["exported_weights"] = str(exported)

    config["finished_at"] = datetime.now(timezone.utc).isoformat()
    config["best_weights"] = str(best)
    config["last_weights"] = str(run_dir / "weights" / "last.pt")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved config: {run_dir / 'train_config.json'}")
    print(f"Best weights: {config['best_weights']}")
    return run_dir


def main() -> int:
    args = parse_args()
    run_dir = train(args)
    from scripts.evaluate import evaluate_run
    from scripts.evaluate_robustness import evaluate_robustness

    evaluate_run(run_dir, args.data, detect_device(args.device), args.imgsz)
    evaluate_robustness(run_dir / "weights" / "best.pt", args.data, detect_device(args.device), args.imgsz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
