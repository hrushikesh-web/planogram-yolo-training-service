import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict

from ultralytics import YOLO


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model: str
    epochs: int
    imgsz: int
    batch: int


def create_dataset_yaml(dataset_dir: str, output_path: str) -> str:
    """
    Create a minimal dataset.yaml for YOLOv8 pointing to the local dataset directory.

    The expected local structure is:
      dataset/
        images/train
        images/val
        labels/train
        labels/val
    """
    import yaml

    images_train = os.path.join(dataset_dir, "images", "train")
    images_val = os.path.join(dataset_dir, "images", "val")

    data = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "product"},
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

    logger.info("dataset.yaml created at '%s'.", output_path)
    return output_path


def train_yolo_model(
    training_config: TrainingConfig,
    dataset_yaml_path: str,
    runs_dir: str = "runs/detect",
) -> str:
    """
    Train a YOLOv8 model using the provided configuration and dataset definition.

    :param training_config: TrainingConfig with model, epochs, imgsz, batch.
    :param dataset_yaml_path: Path to dataset.yaml for YOLOv8.
    :param runs_dir: Directory where YOLO will write runs (default 'runs/detect').
    :return: Path to the best weights file (best.pt).
    """
    logger.info(
        "Starting YOLOv8 training: model=%s, epochs=%d, imgsz=%d, batch=%d",
        training_config.model,
        training_config.epochs,
        training_config.imgsz,
        training_config.batch,
    )

    if not os.path.isfile(dataset_yaml_path):
        raise FileNotFoundError(f"dataset.yaml not found at '{dataset_yaml_path}'")

    os.makedirs(runs_dir, exist_ok=True)

    try:
        model = YOLO(training_config.model)
    except Exception as exc:
        logger.exception("Failed to load YOLO model '%s'.", training_config.model)
        raise RuntimeError(
            f"Unable to load YOLO model '{training_config.model}'"
        ) from exc

    try:
        results = model.train(
            data=dataset_yaml_path,
            epochs=training_config.epochs,
            imgsz=training_config.imgsz,
            batch=training_config.batch,
            project="runs/detect",
            name="train",
        )
    except Exception as exc:
        logger.exception("YOLO training failed.")
        raise RuntimeError("YOLO training failed") from exc

    # Ultralytics stores best weights under runs/detect/train/weights/best.pt by default
    best_model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
    if not os.path.isfile(best_model_path):
        logger.warning(
            "Expected best model file not found at '%s'. Attempting to infer from results.",
            best_model_path,
        )
        # Fallback: try to inspect results if available
        candidate = _infer_best_model_from_results(results)
        if candidate:
            best_model_path = candidate

    if not os.path.isfile(best_model_path):
        msg = f"best.pt not found after training. Expected at '{best_model_path}'."
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Training completed. Best model saved at '%s'.", best_model_path)
    return best_model_path


def _infer_best_model_from_results(results: Any) -> str:
    """
    Try to infer the location of best.pt from the results object.
    This is defensive in case defaults change in future versions.
    """
    try:
        result0 = None
        if isinstance(results, list) and results:
            result0 = results[0]
        elif results is not None:
            result0 = results

        if result0 is not None:
            save_dir = getattr(result0, "save_dir", None)
            if save_dir:
                candidate = os.path.join(str(save_dir), "weights", "best.pt")
                if os.path.isfile(candidate):
                    return candidate
    except Exception:
        logger.debug("Unable to infer best model path from results.", exc_info=True)

    # Defensive fallback: pick the newest best.pt under runs/detect.
    try:
        candidates = list(Path("runs/detect").glob("**/weights/best.pt"))
        if candidates:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            return str(newest)
    except Exception:
        logger.debug("Unable to infer best model path from filesystem.", exc_info=True)

    return ""
