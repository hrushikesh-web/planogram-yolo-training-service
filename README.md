## Planogram YOLO Training Service

This repository provides a **production-ready training pipeline** for Ultralytics YOLOv8 to detect shelf products.  
It is designed to run on a GPU-enabled VM and integrates with **Google Cloud Storage (GCS)** for dataset input and trained model output.

After cloning the repo and installing dependencies, you can run the end-to-end pipeline with:

```bash
bash scripts/run_training_pipeline.sh
```

---

### Requirements

- **Python**: 3.10+
- **GPU**: NVIDIA GPU with recent drivers
- **CUDA / cuDNN**: Matching your PyTorch build
- **GCS access**:
  - `GOOGLE_APPLICATION_CREDENTIALS` environment variable set to a valid service account JSON
  - Or workload identity / default credentials configured on the VM

---

### Installation

1. **Clone the repository**

```bash
git clone <YOUR_REPO_URL> planogram-yolo-training-service
cd planogram-yolo-training-service
```

2. **(Recommended) Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install Python dependencies**

> On GPU VMs, make sure you select a PyTorch build that matches your CUDA version.  
> The `requirements.txt` file includes `torch`, but for maximum control you may want to install the official wheel from the [PyTorch website](https://pytorch.org) before installing the remaining requirements.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Configuration

All configuration for datasets, GCS buckets, and training parameters lives in `config.yaml`.

Example:

```yaml
dataset_bucket: planogram-ml-datasets
model_bucket: planogram-ml-models
dataset_version: v1

training:
  model: yolov8n.pt
  epochs: 50
  imgsz: 640
  batch: 16
```

- **dataset_bucket**: GCS bucket that contains datasets.
- **model_bucket**: GCS bucket used to store trained models.
- **dataset_version**: Dataset version folder under `gs://<dataset_bucket>/datasets/`.
- **training.model**: YOLOv8 model name or path (e.g. `yolov8n.pt`, `yolov8s.pt`, or a custom checkpoint).
- **training.epochs**: Number of training epochs.
- **training.imgsz**: Training image size (single integer).
- **training.batch**: Training batch size.

You can edit `config.yaml` directly to match your environment.

---

### Expected Dataset Layout in GCS

The dataset is expected to be stored in GCS as:

```text
gs://<dataset_bucket>/datasets/<dataset_version>/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/
```

Each `labels/*.txt` file should follow standard YOLO format.

When you run the pipeline, the dataset is downloaded locally into the `dataset/` directory with the same structure:

```text
dataset/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/
```

The training script will auto-generate a `dataset.yaml` file that points to this local structure.

---

### Running the Training Pipeline

Ensure your GCS credentials are configured (for example):

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

Then run:

```bash
bash scripts/run_training_pipeline.sh
```

This will:

1. **Download** the dataset from GCS to `./dataset/`.
2. **Train** the YOLOv8 model using Ultralytics.
3. **Upload** the best model weights (`best.pt`) back to GCS at:

```text
gs://<model_bucket>/yolo/<dataset_version>/best.pt
```

Logs and training artifacts are stored under `runs/detect/train/` by default.

---

### Running Individual Steps Manually

- **Download dataset only**

```bash
python scripts/download_dataset.py
```

- **Train model only**

```bash
python scripts/train_model.py
```

- **Upload model only**

```bash
python scripts/upload_model.py
```

---

### Running on a GPU VM

1. Provision a VM with:
   - An NVIDIA GPU
   - CUDA and cuDNN installed
   - Appropriate NVIDIA drivers
2. Install Python 3.10+.
3. Install PyTorch with GPU support (matching CUDA).
4. Install the rest of the dependencies via:

```bash
pip install -r requirements.txt
```

5. Ensure GCS credentials are available (e.g. via `GOOGLE_APPLICATION_CREDENTIALS`).
6. Run:

```bash
bash scripts/run_training_pipeline.sh
```

---

### Logging and Error Handling

- All scripts use Python's `logging` module for structured logs.
- Failures in dataset download, training, or model upload will:
  - Log an error message, and
  - Exit with a non-zero status code so CI/automation can detect failures.

---

### Notes

- The repository does **not** include actual dataset files or trained models.  
- Make sure your GCS buckets and IAM permissions are correctly configured before running the pipeline.

# planogram-yolo-training-service
# planogram-yolo-training-service
