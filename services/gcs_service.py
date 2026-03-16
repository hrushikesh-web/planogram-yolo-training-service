import logging
import os
from typing import Optional

from google.cloud import storage


logger = logging.getLogger(__name__)


def _get_storage_client() -> storage.Client:
    """
    Returns a Google Cloud Storage client.

    Relies on standard authentication mechanisms:
    - GOOGLE_APPLICATION_CREDENTIALS
    - or default credentials on GCE/GKE, etc.
    """
    try:
        client = storage.Client()
        logger.debug("Initialized Google Cloud Storage client.")
        return client
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to initialize Google Cloud Storage client.")
        raise RuntimeError("Unable to initialize Google Cloud Storage client") from exc


def download_dataset(bucket_name: str, dataset_version: str, local_path: str) -> None:
    """
    Download a dataset from GCS to a local directory.

    Expected layout in GCS:
      gs://<bucket_name>/datasets/<dataset_version>/**

    All objects under this prefix will be mirrored under local_path.
    """
    logger.info(
        "Starting dataset download from GCS bucket '%s', version '%s' into '%s'",
        bucket_name,
        dataset_version,
        local_path,
    )

    if not bucket_name:
        raise ValueError("bucket_name must not be empty")
    if not dataset_version:
        raise ValueError("dataset_version must not be empty")

    client = _get_storage_client()
    try:
        bucket = client.bucket(bucket_name)
    except Exception as exc:
        logger.exception("Failed to access bucket '%s'.", bucket_name)
        raise RuntimeError(f"Unable to access bucket '{bucket_name}'") from exc

    prefix = f"datasets/{dataset_version}/"
    blobs = list(client.list_blobs(bucket, prefix=prefix))

    if not blobs:
        msg = (
            f"No dataset objects found in 'gs://{bucket_name}/{prefix}'. "
            "Check bucket name and dataset_version."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    for blob in blobs:
        relative_path = blob.name[len(prefix) :]
        if not relative_path:
            continue  # skip the directory placeholder, if any

        dest_path = os.path.join(local_path, relative_path)
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)

        logger.debug("Downloading 'gs://%s/%s' to '%s'", bucket_name, blob.name, dest_path)
        try:
            blob.download_to_filename(dest_path)
        except Exception as exc:
            logger.exception(
                "Failed to download object '%s' from bucket '%s'.", blob.name, bucket_name
            )
            raise RuntimeError(
                f"Failed to download object '{blob.name}' from bucket '{bucket_name}'"
            ) from exc

    logger.info("Dataset download completed successfully into '%s'.", local_path)


def upload_model(
    bucket_name: str, model_path: str, destination_path: str, content_type: Optional[str] = None
) -> None:
    """
    Upload a trained model file to GCS.

    :param bucket_name: Target GCS bucket name.
    :param model_path: Local path to model file (e.g. 'runs/detect/train/weights/best.pt').
    :param destination_path: Destination object path within the bucket
                             (e.g. 'yolo/v1/best.pt').
    :param content_type: Optional content type for the uploaded object.
    """
    logger.info(
        "Uploading model from '%s' to 'gs://%s/%s'",
        model_path,
        bucket_name,
        destination_path,
    )

    if not os.path.isfile(model_path):
        msg = f"Model file '{model_path}' does not exist."
        logger.error(msg)
        raise FileNotFoundError(msg)

    if not bucket_name:
        raise ValueError("bucket_name must not be empty")
    if not destination_path:
        raise ValueError("destination_path must not be empty")

    client = _get_storage_client()
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_path)
    except Exception as exc:
        logger.exception("Failed to prepare upload to bucket '%s'.", bucket_name)
        raise RuntimeError(f"Unable to access bucket '{bucket_name}'") from exc

    try:
        blob.upload_from_filename(model_path, content_type=content_type)
    except Exception as exc:
        logger.exception(
            "Failed to upload model '%s' to 'gs://%s/%s'.", model_path, bucket_name, destination_path
        )
        raise RuntimeError(
            f"Failed to upload model '{model_path}' to 'gs://{bucket_name}/{destination_path}'"
        ) from exc

    logger.info(
        "Successfully uploaded model '%s' to 'gs://%s/%s'.",
        model_path,
        bucket_name,
        destination_path,
    )

