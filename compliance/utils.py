import gzip
import io
from botocore.exceptions import ClientError
from typing import Optional
import zlib
from compliance.config import s3_client, BUCKET_NAME


def ensure_test_bucket_exists():
    # Create required bucket if it doesn't yet exist
    try:
        s3_client.create_bucket(Bucket=BUCKET_NAME)
    except ClientError:
        pass  # Bucket already exists


def upload_to_s3(s3_client, data: bytes, filename: str) -> None:
    """Upload a some binary data to an S3 storage bucket"""

    stream = io.BytesIO(data)
    s3_client.upload_fileobj(stream, BUCKET_NAME, filename)

    return


def fetch_from_s3(s3_client, filename: str) -> bytes:
    """Fetches data from configured S3 source and returns the content as raw bytes"""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=filename)
        content = response["Body"].read()
        return content
    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(
            f"File '{filename}' not found in S3 bucket '{BUCKET_NAME}'"
        )


def filter_pipeline(data: bytes, compression: Optional[str]) -> bytes:
    """Apply compression and filters to data and return the result."""
    if compression == "gzip":
        data = gzip.compress(data)
    elif compression == "zlib":
        data = zlib.compress(data)
    elif compression is not None:
        raise AssertionError(f"Unexpected compression algorithm {compression}")
    return data
