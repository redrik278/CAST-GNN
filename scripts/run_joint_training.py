#!/usr/bin/env python

import argparse
from cast_gnn_project.src.utils import load_config, setup_logging, merge_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run joint training of CAST‑GNN")
    parser.add_argument("--eegmmidb_config", type=str, default="configs/eegmmidb.yaml", help="Path to EEGMMIDB YAML")
    parser.add_argument("--milimbeeg_config", type=str, default="configs/milimbeeg.yaml", help="Path to MILimbEEG YAML")
    parser.add_argument("--joint_config", type=str, default="configs/joint.yaml", help="Path to joint training YAML")
    args = parser.parse_args()
    eeg_cfg = load_config(args.eegmmidb_config)
    mil_cfg = load_config(args.milimbeeg_config)
    joint_cfg = load_config(args.joint_config)
    cfg = merge_configs(eeg_cfg, mil_cfg, joint_cfg)
    logger = setup_logging("outputs/logs", log_name="run_joint_training.log")
    logger.info("Merged configuration: %s", cfg)
    # TODO: implement joint training in later phases
    logger.info("Joint training placeholder executed.")


if __name__ == "__main__":
    main()