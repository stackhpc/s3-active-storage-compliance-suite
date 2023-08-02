import json
import math
import numpy as np
import numpy.ma as ma
import os
import pytest
import requests
import sys
from typing import List, Union

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
    FILTER_ALGS,
    MISSING_DATA,
    TEST_BYTE_ORDER,
)
from .missing import Missing, ValidMax, ValidMin
from .mocks import MockResponse
from .utils import filter_pipeline, ensure_test_bucket_exists, upload_to_s3


def generate_test_array(
    dtype: str,
    shape: list[int],
    size: Union[int, None],
    missing: Union[Missing, None],
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
    data = (10 * np.random.rand(num_elements)).astype(dtype)
    # print("Raw\n", data)

    if missing:
        # Mark some data as missing.
        data = missing.make_holes(data)
        # print("Masked\n", data)

    return data


def generate_object_data(
    data,
    offset: Union[int, None],
    trailing: Union[int, None],
    compression: Union[str, None],
    filters: Union[List[str], None],
    dtype: str,
    byte_order: Union[str, None],
):
    """
    Generate S3 object data from a numpy array.
    Applies an offset, and trailing data before upload.
    Returns a 2-tuple of:
      * the object data
      * the size of the data in bytes after application of the filter pipeline
    """
    if byte_order and byte_order != sys.byteorder:
        # Swap the byte order of the underlying data.
        # Unsure why data.newbyteorder() doesn't work.
        data = data.byteswap()

    # Convert to bytes for upload
    data_bytes = data.tobytes()
    print(data_bytes)
    # assert False

    # Apply the compression and filter pipeline.
    element_size = np.dtype(dtype).itemsize
    filtered_data = filter_pipeline(data_bytes, compression, filters, element_size)

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


def perform_operation(data, operation):
    return OPERATION_FUNCS[operation](data)


def calculate_expected_result(data, operation, shape, selection, order, missing):
    """
    Calculate the expected result from applying the operation to the data.
    Returns the result as a numpy array or scalar.
    """
    # Reshape the array to apply the shape (if specified) and C/F order.
    data = data.reshape(*(shape or data.shape), order=order)

    if missing:
        unmasked_result = perform_operation(data, operation)
        data = missing.mask(data)

    # Create pythonic slices object
    # (must be a tuple of slice objects for multi-dimensional indexing of numpy arrays)
    if selection:
        unselected_result = perform_operation(data, operation)
        slices = tuple(slice(*s) for s in selection)
        data = data[slices]

    # Perform main operation
    operation_result = perform_operation(data, operation)

    # Verify that the parameters affect the result, so that we can verify
    # that they've been applied.

    # A select result is not affected by missing data.
    if missing and operation != "select":
        assert not np.array_equal(operation_result, unmasked_result)

    # A min/max result may not be affected by a selection.
    if selection and operation not in ["min", "max"]:
        assert not np.array_equal(operation_result, unselected_result)

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
    filters: Union[List[str], None] = None,
    missing: Union[Missing, None] = None,
    byte_order: Union[str, None] = None,
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
    data = generate_test_array(dtype, shape, size, missing)

    # Generate S3 object data from the array
    object_data, compressed_size = generate_object_data(
        data,
        offset,
        trailing,
        compression,
        filters,
        dtype,
        byte_order,
    )

    # Create an object in S3
    create_test_s3_object(object_data, filename)

    # Calculate and return the expected result.
    data, operation_result = calculate_expected_result(
        data,
        operation,
        shape,
        selection,
        order,
        missing,
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
    filters=None,
    missing=None,
    byte_order=None,
):
    """Test basic functionality of reduction operations on various types of input data"""

    filename = f"test--operation-{operation}-dtype-{dtype}--shape-{shape}-selection-{selection}-order-{order}-offset-{offset}-size-{size}-trailing-{trailing}-compression-{compression}-filters-{filters}-missing-{missing}-byte-order-{byte_order}.bin"
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
        filters,
        missing,
        byte_order,
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
    if filters:
        request_data["filters"] = [
            {"id": filter, "element_size": np.dtype(dtype).itemsize}
            for filter in filters
        ]
    if missing:
        request_data["missing"] = missing.to_request_data()
    if byte_order:
        request_data["byte_order"] = byte_order

    # print(request_data)

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
        assert proxy_response.headers["x-activestorage-count"] == str(
            ma.count(array_data)
        )
    if TEST_BYTE_ORDER:
        assert proxy_response.headers["x-activestorage-byte-order"] == "little"
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
@pytest.mark.parametrize("filter", [""] + FILTER_ALGS)
@pytest.mark.parametrize("offset", [None, 64])
@pytest.mark.parametrize("trailing", [None, 64])
def test_compression(monkeypatch, operation, offset, trailing, compression, filter):
    """
    Test compressed and/or filtered data with and without an offset and trailing data.
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
        filters=[filter] if filter else None,
    )


@pytest.mark.parametrize("dtype", ALLOWED_DTYPES)
@pytest.mark.parametrize("operation", OPERATION_FUNCS.keys())
@pytest.mark.parametrize(
    "selection",
    [
        None,
        [[0, 10, 2], [0, 2, 1], [0, 3, 1]],
    ],
)
@pytest.mark.parametrize("missing_cls", MISSING_DATA)
def test_missing_data(monkeypatch, dtype, operation, selection, missing_cls):
    """
    Test datasets with missing data.
    """
    # Certain operations are not easily tested, so skip them.
    invalid = [
        ("min", ValidMax),
        ("max", ValidMin),
    ]
    if (operation, missing_cls) in invalid:
        pytest.skip(f"Can't test operation {operation} with missing {missing_cls}")

    # Create an appropriate missing data description for this dtype and operation.
    missing = missing_cls.create(np.dtype(dtype), operation)

    test_basic_operation(
        monkeypatch,
        operation,
        dtype,
        [10, 5, 2],
        selection,
        "C",
        missing=missing,
    )


@pytest.mark.skipif(not TEST_BYTE_ORDER, reason="Byte order not supported")
@pytest.mark.parametrize("dtype", ALLOWED_DTYPES)
@pytest.mark.parametrize("operation", OPERATION_FUNCS.keys())
@pytest.mark.parametrize(
    "selection",
    [
        None,
        [[0, 10, 2], [0, 2, 1], [0, 3, 1]],
    ],
)
@pytest.mark.parametrize("byte_order", ["little", "big"])
def test_byte_order(monkeypatch, dtype, operation, selection, byte_order):
    """
    Test datasets with different byte orders.
    """
    test_basic_operation(
        monkeypatch,
        operation,
        dtype,
        [10, 5, 2],
        selection,
        "C",
        byte_order=byte_order,
    )
