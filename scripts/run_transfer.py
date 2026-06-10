#!/usr/bin/env python

import argparse
from cast_gnn_project.src.utils import load_config, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross‑dataset transfer experiments")
    parser.add_argument("--source_config", type=str, required=True, help="Path to source dataset configuration YAML")
    parser.add_argument("--target_config", type=str, required=True, help="Path to target dataset configuration YAML")
    parser.add_argument("--joint_config", type=str, default="configs/joint.yaml", help="Path to joint training configuration YAML")
    args = parser.parse_args()
    src_cfg = load_config(args.source_config)
    tgt_cfg = load_config(args.target_config)
    joint_cfg = load_config(args.joint_config)
    logger = setup_logging("outputs/logs", log_name="run_transfer.log")
    logger.info("Loaded source configuration: %s", src_cfg)
    logger.info("Loaded target configuration: %s", tgt_cfg)
    logger.info("Loaded joint configuration: %s", joint_cfg)
    # TODO: implement transfer training and evaluation in later phases
    logger.info("Cross‑dataset transfer placeholder executed.")


if __name__ == "__main__":
    main()