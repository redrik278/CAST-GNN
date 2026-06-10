#!/usr/bin/env python

import argparse
from cast_gnn_project.src.utils import load_config, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate robustness of CAST‑GNN")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logging("outputs/logs", log_name="run_robustness.log")
    logger.info("Loaded configuration: %s", cfg)
    # TODO: implement robustness testing in later phases
    logger.info("Robustness evaluation placeholder executed.")


if __name__ == "__main__":
    main()