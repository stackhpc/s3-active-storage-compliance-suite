# S3 active storage compliance suite

Integration tests and performance benchmarking tools for implementations of S3 Active Storage servers.

There are currently two S3 active storage server implementations:

* [Reductionist](https://github.com/stackhpc/reductionist-rs) - a performant server implementation
* [S3 Active Storage Prototype](https://github.com/stackhpc/s3-active-storage-prototype) - a functional prototype

## Compliance Suite Usage

To set up the testing environment run:

```
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

To run the compliance test suite on your own implementation of an S3 active storage server, edit the following variables in `compliance/config.py`:

- `S3_SOURCE` - The address of your S3 store (e.g. `https://s3-proxy.com/`). If you don't have an existing S3 store you can set up a temporary minio docker container by running `scripts/run-minio.sh` in a separate terminal, in which case you should leave the S3 source as localhost.
  
- `AWS_ID` - The AWS access key ID used to authenticate with the S3 source (defaults to 'minioadmin' for the minio docker container).

- `AWS_PASSWORD` - The secret access key used to authenticate with the S3 source (defaults to 'minioadmin' for the minio docker container)

- `PROXY_URL` - The address for your active storage proxy implementation (e.g. `https://s3-proxy.example.com/`)

The compliance test suite can then be run by calling 
```
pytest
```
from within the project directory.

### Testing older active storage servers

We aim to add tests for features as they are added to the S3 active storage server.
This does lead to problems when testing older versions of the server that may lack support for those features.
This is addressed through configuration variables in `compliance/config.py`.

- `TEST_X_ACTIVESTORAGE_COUNT_HEADER` - Whether to test for the presence of the `x-activestorage-count` header in responses.
- `COMPRESSION_ALGS` - List of names of compression algorithms to test. May be set to an empty list.
- `FILTER_ALGS` - List of names of filter algorithms to test. May be set to an empty list.

### Implementation details

Test data is currently generated as numpy arrays and then uploaded to the configured S3 source in binary format. Following this upload, requests are made to the active storage proxy and the proxy response is compared to the expected result based on the agreed API specification and the generated test arrays.

There are procedurally generated test cases to cover various combinations of reduction operation, data type, data shape and data slice parameters as well as testing of response codes and error messages informative error messages.



## Performance Benchmarking

The `scripts/run-benchmarks.py` file can be used to benchmark running instances of the active storage proxy. To do so, edit the required config parameters at the top of the script (`PROXY_URLS`, `S3_SOURCE` & `AUTH`) and then run the script. 

The benchmarking process will generate a few numpy arrays then upload them to a test bucket within the configured S3 source. The `sum` operation is then called repeatedly on each of these arrays to collect some timing statistics. For convenience, the script will also collect timing stats while *bypassing* the active storage proxy (i.e. fetching the full array from S3 and then performing the sum locally) to indicate how much of an improvement the active storage proxy is providing. Once all benchmark runs are finished, a boxplot figure will be generated (at the path `./benchmark--{timestamp}.png`) to summarize the benchmark results. Finally, all generated test data is removed from the S3 source.

## Contributing

The following checks are run in CI (from the repository root):

```
black .
mypy compliance
flake8
```
