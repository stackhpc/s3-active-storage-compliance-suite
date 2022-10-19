# S3 active storage compliance suite

Unit tests and performance benchmarking tools for implementations of the s3-active-storage project.

## Outline 

The working plan for this repo is that, since we don't have an up-to-date reference implementation which conforms to the recently revised API spec, the default behaviour initially will be to mock responses from the active storage proxy. These mocks can be replaced by a URL pointing to a functional implementation of the proxy service at a later date.

---

## Compliance Suite Usage

To set up the testing environment run:

```
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

To run the compliance test suite on your own implementation of an S3 active storage client, edit the following variables in `compliance/config.py`:

- `S3_SOURCE` - The address of your S3 store (e.g. `https://s3-proxy.com/`). If you don't have an existing S3 store you can set up a temporary minio docker container by running `scripts/run-minio.sh` in a separate terminal, in which case you should leave the S3_SOURCE as localhost.
  
- `AWS_ID` - The AWS access key ID used to authenticate with the S3 source (defaults to 'minioadmin' for the minio docker container).

- `AWS_PASSWORD` - The secret access key used to authenticate with the S3 source (defaults to 'minioadmin' for the minio docker container)

- `PROXY_URL` - The address for your active storage proxy implementation (e.g. `https://s3-proxy.example.com/`)

The compliance test suite can the be running by calling 
```
pytest
```
from within the project directory.

### Implementation details

Test data is currently generated as numpy arrays and then uploaded to the configured S3 source in binary format. Following this upload, requests are made to active storage proxy and the proxy response is compared to the expected result based on the generated test data and the agreed API specification.

There are procedurally generated test cases to cover various combinations of reduction operation, data type, data shape and data slice specifications. Testing of other aspects (e.g. response codes and error messages) can be added as needed.


---

## Performance Benchmarking

_Coming soon..._
