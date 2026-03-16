import logging
import os
import sys
from pathlib import Path

import yaml

from services.gcs_service import download_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("download_dataset")


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must define a YAML mapping.")

    return config


def main() -> None:
    try:
        config = load_config()
        dataset_bucket = config.get("dataset_bucket")
        dataset_version = config.get("dataset_version")

        if not dataset_bucket or not dataset_version:
            raise ValueError(
                "Both 'dataset_bucket' and 'dataset_version' must be specified in config.yaml."
            )

        local_dataset_dir = Path("dataset")
        local_dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading dataset version '%s' from bucket '%s' into '%s'",
            dataset_version,
            dataset_bucket,
            local_dataset_dir,
        )

        download_dataset(
            bucket_name=dataset_bucket,
            dataset_version=dataset_version,
            local_path=str(local_dataset_dir),
        )

        logger.info("Dataset downloaded successfully.")
    except Exception as exc:
        logger.exception("Dataset download failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

