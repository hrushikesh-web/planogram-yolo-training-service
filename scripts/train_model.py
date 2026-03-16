import logging
import os
import sys
from pathlib import Path

import yaml

from services.yolo_training_service import TrainingConfig, create_dataset_yaml, train_yolo_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_model")


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
        training_cfg = config.get("training") or {}

        model_name = training_cfg.get("model")
        epochs = int(training_cfg.get("epochs", 50))
        imgsz = int(training_cfg.get("imgsz", 640))
        batch = int(training_cfg.get("batch", 16))

        if not model_name:
            raise ValueError("training.model must be specified in config.yaml.")

        dataset_dir = Path("dataset")
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory '{dataset_dir}' not found. "
                "Run 'python scripts/download_dataset.py' first."
            )

        dataset_yaml_path = Path("dataset.yaml")
        create_dataset_yaml(dataset_dir=str(dataset_dir), output_path=str(dataset_yaml_path))

        training_config = TrainingConfig(
            model=model_name,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
        )

        best_model_path = train_yolo_model(
            training_config=training_config,
            dataset_yaml_path=str(dataset_yaml_path),
        )

        # Persist best model path for the upload step
        metadata_path = Path("models")
        metadata_path.mkdir(parents=True, exist_ok=True)
        best_txt = metadata_path / "best_model_path.txt"
        best_txt.write_text(best_model_path, encoding="utf-8")

        logger.info("Training completed. Best model at: %s", best_model_path)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

