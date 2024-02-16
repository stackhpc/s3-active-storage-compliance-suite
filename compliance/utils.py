import gzip
import io
import json
from botocore.exceptions import ClientError

# NOTE: numcodecs is missing type hints
import numcodecs  # type: ignore
from typing import Optional
import zlib
from compliance.config import s3_client, BUCKET_NAME, PUBLIC_BUCKET_NAME


def get_bucket_name(public: bool = False) -> str:
    return PUBLIC_BUCKET_NAME if public else BUCKET_NAME


def delete_bucket(public: bool):
    # Currently unused, provided in case it's useful.
    bucket = get_bucket_name(public)
    try:
        objs = s3_client.list_objects_v2(Bucket=bucket)
        for obj in objs.get("Contents", []):
            s3_client.delete_object(Bucket=bucket, Key=obj["Key"])
        s3_client.delete_bucket(Bucket=bucket)
    except ClientError:
        pass  # No bucket


def ensure_test_bucket_exists(public: bool = False):
    bucket = get_bucket_name(public)
    # Create required bucket if it doesn't yet exist
    try:
        s3_client.create_bucket(Bucket=bucket)
    except ClientError:
        pass  # Bucket already exists

    if public:
        apply_public_policy()


def apply_public_policy():
    # Apply a policy that allows unauthenticated read access to objects in the bucket.
    bucket = get_bucket_name(True)
    policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": ["s3:GetObject"],
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Resource": [f"arn:aws:s3:::{bucket}/*"],
                    "Sid": "",
                }
            ],
        }
    )
    s3_client.put_bucket_policy(Bucket=bucket, Policy=policy)


def upload_to_s3(s3_client, data: bytes, filename: str, public: bool = False) -> None:
    """Upload a some binary data to an S3 storage bucket"""

    bucket = get_bucket_name(public)
    stream = io.BytesIO(data)
    s3_client.upload_fileobj(stream, bucket, filename)

    return


def fetch_from_s3(s3_client, filename: str, public: bool = False) -> bytes:
    """Fetches data from configured S3 source and returns the content as raw bytes"""
    bucket = get_bucket_name(public)
    try:
        response = s3_client.get_object(Bucket=bucket, Key=filename)
        content = response["Body"].read()
        return content
    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"File '{filename}' not found in S3 bucket '{bucket}'")


def filter_pipeline(
    data: bytes, compression: Optional[str], filters: Optional[list], element_size: int
) -> bytes:
    """Apply compression and filters to data and return the result."""
    for filter in filters or []:
        if filter == "shuffle":
            data = numcodecs.Shuffle(element_size).encode(data)
        else:
            raise AssertionError(f"Unexpected filter algorithm {filter}")
    if compression == "gzip":
        data = gzip.compress(data)
    elif compression == "zlib":
        data = zlib.compress(data)
    elif compression is not None:
        raise AssertionError(f"Unexpected compression algorithm {compression}")
    return data
