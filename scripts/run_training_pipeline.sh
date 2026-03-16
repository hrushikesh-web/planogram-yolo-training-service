#!/usr/bin/env bash
set -euo pipefail

echo "Downloading dataset from GCS..."
python scripts/download_dataset.py

echo "Training YOLO model..."
python scripts/train_model.py

echo "Uploading trained model..."
python scripts/upload_model.py

echo "Training pipeline completed."

