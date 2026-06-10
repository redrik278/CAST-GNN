"""Unified command-line entry point for CAST-GNN.

Examples
--------
python -m src.main --mode train_within --dataset eegmmidb --task B_execution_vs_imagery --config configs/eegmmidb.yaml
python -m src.main --mode train_transfer --source eegmmidb --target milimbeeg --task B_execution_vs_imagery --config configs/transfer.yaml
python -m src.main --mode train_joint --task B_execution_vs_imagery --config configs/joint.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.training.train_joint import run_joint_training
from src.training.train_transfer import run_transfer_training
from src.training.train_within_dataset import run_within_dataset_training
from src.utils import load_many_configs, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CAST-GNN command-line interface")
    parser.add_argument("--mode", required=True, choices=["train_within", "train_transfer", "train_joint"], help="Execution mode")
    parser.add_argument("--config", nargs="+", required=True, help="One or more YAML config files")
    parser.add_argument("--dataset", default=None, help="Dataset for within-dataset training")
    parser.add_argument("--source", default=None, help="Source dataset for transfer")
    parser.add_argument("--target", default=None, help="Target dataset for transfer")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--no_graph_adapters", action="store_true", help="Disable graph adapters in joint mode")
    parser.add_argument("--no_coral", action="store_true", help="Disable CORAL alignment in joint mode")
    return parser.parse_args()


def main() -> None:
    """Run the selected pipeline mode."""
    args = parse_args()
    cfg: dict[str, Any] = load_many_configs(*args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)

    if args.mode == "train_within":
        if args.dataset is None:
            raise ValueError("--dataset is required for train_within mode.")
        metrics = run_within_dataset_training(args.dataset, cfg, args.task, seed=seed)

    elif args.mode == "train_transfer":
        if args.source is None or args.target is None:
            raise ValueError("--source and --target are required for train_transfer mode.")
        metrics = run_transfer_training(args.source, args.target, cfg, args.task, seed=seed)

    elif args.mode == "train_joint":
        metrics = run_joint_training(
            cfg,
            task_name=args.task,
            seed=seed,
            use_graph_adapters=not args.no_graph_adapters,
            use_coral=not args.no_coral,
        )

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print("Final metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
