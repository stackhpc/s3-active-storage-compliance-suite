import json
import math
import os
import pytest
import requests
import numpy as np

from typing import Union
from .config import (
    s3_client,
    S3_SOURCE,
    PROXY_URL,
    BUCKET_NAME,
    ALLOWED_DTYPES,
    OPERATION_FUNCS,
    AWS_ID,
    AWS_PASSWORD,
    TEST_X_ACTIVESTORAGE_COUNT_HEADER,
    COMPRESSION_ALGS,
)
from .utils import filter_pipeline, ensure_test_bucket_exists, upload_to_s3
from .mocks import MockResponse


def generate_test_array(
    dtype: str,
    shape: list[int],
    size: Union[int, None],
):
    """
    Generate and return a numpy array of random data.
    """
    np.random.seed(10)  # Make sure randomized arrays are reproducible

    # Determine the number of elements in the array.
    if shape:
        num_elements = math.prod(shape)
        if size:
            assert num_elements == size // np.dtype(dtype).itemsize
    elif size:
        num_elements = size // np.dtype(dtype).itemsize
    else:
        num_elements = 100

    # Generate some test data
    # This is the raw data that will be uploaded to S3, and is currently a 1D array.
    # (multiply random array by 10 so that int dtypes don't all round down to zeros)
    return (10 * np.random.rand(num_elements)).astype(dtype)


def generate_object_data(
    data,
    offset: Union[int, None],
    trailing: Union[int, None],
    compression: Union[str, None],
):
    """
    Generate S3 object data from a numpy array.
    Applies an offset, and trailing data before upload.
    Returns a 2-tuple of:
      * the object data
      * the size of the data in bytes after application of the filter pipeline
    """
    # Convert to bytes for upload
    data_bytes = data.tobytes()

    # Apply the compression and filter pipeline.
    filtered_data = filter_pipeline(data_bytes, compression)

    # Apply random data before offset.
    object_data = os.urandom(offset or 0) + filtered_data + os.urandom(trailing or 0)
    return object_data, len(filtered_data)


def create_test_s3_object(
    object_data,
    filename,
):
    """
    Create an S3 object from a list of bytes.
    """
    # Add data to s3 bucket so that proxy can use it
    ensure_test_bucket_exists()
    upload_to_s3(s3_client, object_data, filename)


def calculate_expected_result(data, operation, shape, selection, order):
    """
    Calculate the expected result from applying the operation to the data.
    Returns the result as a numpy array or scalar.
    """
    # Reshape the array to apply the shape (if specified) and C/F order.
    data = data.reshape(*(shape or data.shape), order=order)

    # Create pythonic slices object
    # (must be a tuple of slice objects for multi-dimensional indexing of numpy arrays)
    if selection:
        slices = tuple(slice(*s) for s in selection)
        data = data[slices]

    # Perform main operation
    operation_result = OPERATION_FUNCS[operation](data)

    return data, operation_result


def create_test_data(
    filename,
    operation: str,
    dtype: str,
    shape: list[int],
    selection: list[list[int]],
    offset: Union[int, None],
    size: Union[int, None],
    order: str,
    trailing: Union[int, None] = None,
    compression: Union[str, None] = None,
):
    """
    Creates some test data and uploads it to the configured
    S3 source for later requests through the active proxy
    Returns a 3-tuple containing:
      * a numpy array of the test data
      * the expected result as a numpy array or scalar
      * the size in bytes of the compressed and/or filtered data
    """
    # Generate a test array
    data = generate_test_array(dtype, shape, size)

    # Generate S3 object data from the array
    object_data, compressed_size = generate_object_data(
        data, offset, trailing, compression
    )

    # Create an object in S3
    create_test_s3_object(object_data, filename)

    # Calculate and return the expected result.
    data, operation_result = calculate_expected_result(
        data, operation, shape, selection, order
    )
    return data, operation_result, compressed_size


# Stacking parametrization decorators tells pytest to check every possible combination of parameters
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "shape, selection",
    [
        (None, None),
        ([5, 5, 4], None),
        ([100], [[-10, -50, -4]]),
        ([20, 5], [[0, 19, 2], [1, 3, 1]]),
    ],
)
@pytest.mark.parametrize("dtype", ALLOWED_DTYPES)
@pytest.mark.parametrize("operation", OPERATION_FUNCS.keys())
def test_basic_operation(
    monkeypatch,
    operation,
    dtype,
    shape,
    selection,
    order,
    offset=None,
    size=None,
    trailing=None,
    compression=None,
):
    """Test basic functionality of reduction operations on various types of input data"""

    filename = f"test--operation-{operation}-dtype-{dtype}--shape-{shape}-selection-{selection}-order-{order}-offset-{offset}-size-{size}-trailing-{trailing}-compression-{compression}.bin"
    array_data, operation_result, compressed_size = create_test_data(
        filename,
        operation,
        dtype,
        shape,
        selection,
        offset,
        size,
        order,
        trailing,
        compression,
    )

    request_data = {
        "source": S3_SOURCE,
        "bucket": BUCKET_NAME,
        "object": filename,
        "dtype": dtype,
        "offset": offset,
        "size": compressed_size,
        "shape": shape,
        "order": order,
        "selection": selection,
    }

    if compression:
        request_data["compression"] = {"id": compression}

    # Mock proxy responses if url not set
    if PROXY_URL is None:
        monkeypatch.setattr(
            requests,
            "post",
            lambda *args, **kwargs: MockResponse(
                status_code=200,
                array_data=array_data,
                operation_result=operation_result,
                order=order,
            ),
        )

    # Fetch response from proxy
    proxy_response = requests.post(
        f"{PROXY_URL}/v1/{operation}/", json=request_data, auth=(AWS_ID, AWS_PASSWORD)
    )

    # For debugging failed tests
    if proxy_response.status_code != 200:
        print(proxy_response.text)

    assert proxy_response.status_code == 200

    proxy_result = np.frombuffer(
        proxy_response.content, dtype=proxy_response.headers["x-activestorage-dtype"]
    )

    # Compare to expected result and make sure response headers are sensible - all comparisons should be done as strings
    print(
        "\nProxy result:", proxy_result, "\nExpected result:", operation_result
    )  # For debugging
    assert proxy_response.headers["x-activestorage-dtype"] == (
        request_data["dtype"] if operation != "count" else "int64"
    )
    expected_shape = list(operation_result.shape)
    proxy_shape = json.loads(proxy_response.headers["x-activestorage-shape"])
    assert proxy_shape == expected_shape
    if TEST_X_ACTIVESTORAGE_COUNT_HEADER:
        assert proxy_response.headers["x-activestorage-count"] == str(array_data.size)
    proxy_result = proxy_result.reshape(proxy_shape, order=order)
    assert np.allclose(
        proxy_result, operation_result
    ), f"actual:\n{proxy_result}\n!=\nexpected:\n{operation_result}"
    assert proxy_response.headers["content-length"] == str(
        len(operation_result.tobytes())
    )


# Separate out these tests since valid offset & size values depend on other parameters so combinatorial param approach is too complicated
param_combos = [
    (
        "int64",  # dtype
        [20],  # shape
        None,  # selection
        8 * 10,  # offset
        8 * 20,  # size - must equal product of shape * dtype size in bytes
        None,  # trailing data size in bytes
    ),
    (
        "float32",
        [10, 3],
        [[0, 10, 2], [0, 3, 1]],
        42,
        4 * 30,
        42,
    ),
    (
        "uint32",
        [10, 2, 4],
        [[0, 10, 2], [0, 2, 1], [0, 3, 1]],
        4 * 5,
        4 * 80,
        None,
    ),
    (
        "int32",
        [10, 2, 4],
        [[0, 10, 2], [0, 2, 1], [0, 4, 3]],
        4 * 20,
        None,
        None,
    ),
]


@pytest.mark.parametrize(
    "dtype, shape, selection, offset, size, trailing", param_combos
)
@pytest.mark.parametrize("operation", OPERATION_FUNCS.keys())
@pytest.mark.parametrize("order", ["C", "F"])
def test_offset_and_size(
    monkeypatch, operation, dtype, shape, selection, order, offset, size, trailing
):
    # We can still hook into previous test func though to avoid repeated code
    # (maybe there's a more pytest-y way to do this?)
    test_basic_operation(
        monkeypatch, operation, dtype, shape, selection, order, offset, size, trailing
    )


@pytest.mark.parametrize("operation", OPERATION_FUNCS.keys())
@pytest.mark.parametrize("compression", COMPRESSION_ALGS)
@pytest.mark.parametrize("offset", [None, 64])
@pytest.mark.parametrize("trailing", [None, 64])
def test_compression(monkeypatch, operation, offset, trailing, compression):
    """
    Test compressed data with and without an offset and trailing data.
    """
    test_basic_operation(
        monkeypatch,
        operation,
        "int64",
        [10, 5, 2],
        None,
        "C",
        offset,
        trailing=trailing,
        compression=compression,
    )
