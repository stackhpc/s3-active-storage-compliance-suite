import boto3
import numpy as np
import numpy.ma as ma

from .missing import MissingValue, MissingValues, ValidMax, ValidMin, ValidRange

# Location of upstream S3 data
S3_SOURCE = "http://localhost:9000"
# AWS access key ID
AWS_ID = "minioadmin"
# AWS access key password
AWS_PASSWORD = "minioadmin"
# S3 bucket in which to generate test data
BUCKET_NAME = "active-storage-compliance-test-data"
# Public S3 bucket in which to generate test data
PUBLIC_BUCKET_NAME = "active-storage-compliance-test-data-public"
# Address of active storage proxy to be tested - setting value to 'None' will result in proxy responses being mocked by pytest
PROXY_URL = "http://localhost:8080"
# Optional path to CA cert to use for https requests
PROXY_CA_CERT = None  # '/path/to/certificate.crt'

# Create an S3 client with required credentials
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_SOURCE,
    aws_access_key_id=AWS_ID,
    aws_secret_access_key=AWS_PASSWORD,
)

ALLOWED_DTYPES = ["int32", "int64", "float32", "float64", "uint32", "uint64"]

OPERATION_FUNCS = {
    "select": lambda arr: arr,
    "sum": lambda arr: ma.sum(arr, dtype=arr.dtype),
    "count": lambda arr: np.array(ma.count(arr)),
    "max": np.max,
    "min": np.min,
}

# Whether to test for the presence of the x-activestorage-count header in responses.
TEST_X_ACTIVESTORAGE_COUNT_HEADER = True

# List of names of supported compression algorithms.
# May be set to an empty list if compression is not supported by the server.
COMPRESSION_ALGS = [
    "gzip",
    "zlib",
]

# List of names of supported filter algorithms.
# May be set to an empty list if filters are not supported by the server.
FILTER_ALGS = [
    "shuffle",
]

# List of missing data classes.
# May be set to an empty list if missing data is not supported by the server.
MISSING_DATA = [
    MissingValue,
    MissingValues,
    ValidMax,
    ValidMin,
    ValidRange,
]

# Whether to test data with different byte orders (endianness).
TEST_BYTE_ORDER = True

# Whether to test data stored in publicly accessible buckets.
TEST_PUBLIC_BUCKET = True
