#!/usr/bin/env python
import argparse
from cast_gnn_project.src.utils import load_config, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit EEGMMIDB dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logging("outputs/logs", log_name="audit_eegmmidb.log")
    logger.info("Loaded configuration: %s", cfg)
    # TODO: implement audit logic in later phases
    logger.info("EEGMMIDB audit placeholder executed.")


if __name__ == "__main__":
    main()