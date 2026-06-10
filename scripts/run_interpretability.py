#!/usr/bin/env python

import argparse
from cast_gnn_project.src.utils import load_config, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interpretability analysis for CAST‑GNN")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logging("outputs/logs", log_name="run_interpretability.log")
    logger.info("Loaded configuration: %s", cfg)
    # TODO: implement interpretability analysis in later phases
    logger.info("Interpretability analysis placeholder executed.")


if __name__ == "__main__":
    main()