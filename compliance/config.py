import boto3
import numpy as np

#Location of upstream S3 data
S3_SOURCE = "http://localhost:9000"
#AWS access key ID
AWS_ID = "minioadmin"
#AWS access key password
AWS_PASSWORD = "minioadmin"
#S3 bucket in which to generate test data
BUCKET_NAME = 'active-storage-compliance-test-data'
#Address of active storage proxy to be tested - setting value to 'None' will result in proxy responses being mocked by pytest
PROXY_URL = "http://localhost:8000"

# Create an S3 client with required credentials
s3_client = boto3.client(
    "s3",
    endpoint_url = S3_SOURCE,
    aws_access_key_id = AWS_ID,
    aws_secret_access_key = AWS_PASSWORD,
)

ALLOWED_DTYPES = ['int32', 'int64', 'float32', 'float64', 'uint32', 'uint64']

OPERATION_FUNCS = {
    'select': lambda arr: arr,
    'sum': lambda arr: np.sum(arr, dtype=arr.dtype),
    "count": lambda arr: np.prod(arr.shape, dtype = np.int64),
    'max': np.max,
    'min': np.min,
}
