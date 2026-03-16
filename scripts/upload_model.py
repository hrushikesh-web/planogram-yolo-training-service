import logging
import os
import sys
from pathlib import Path

import yaml

from services.gcs_service import upload_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("upload_model")


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
        model_bucket = config.get("model_bucket")
        dataset_version = config.get("dataset_version")

        if not model_bucket or not dataset_version:
            raise ValueError(
                "Both 'model_bucket' and 'dataset_version' must be specified in config.yaml."
            )

        best_model_meta = Path("models") / "best_model_path.txt"
        if not best_model_meta.is_file():
            raise FileNotFoundError(
                f"Best model path file '{best_model_meta}' not found. "
                "Run 'python scripts/train_model.py' first."
            )

        best_model_path = best_model_meta.read_text(encoding="utf-8").strip()
        if not best_model_path:
            raise ValueError(f"Best model path file '{best_model_meta}' is empty.")

        destination_path = f"yolo/{dataset_version}/best.pt"

        logger.info(
            "Uploading best model '%s' to 'gs://%s/%s'",
            best_model_path,
            model_bucket,
            destination_path,
        )

        upload_model(
            bucket_name=model_bucket,
            model_path=best_model_path,
            destination_path=destination_path,
            content_type="application/octet-stream",
        )

        logger.info("Model upload completed successfully.")
    except Exception as exc:
        logger.exception("Model upload failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

