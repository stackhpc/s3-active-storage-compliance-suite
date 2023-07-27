import numpy as np
import pytest
import requests
import uuid

from .config import (
    s3_client,
    S3_SOURCE,
    BUCKET_NAME,
    PROXY_URL,
    AWS_ID,
    AWS_PASSWORD,
    MISSING_DATA,
)
from .mocks import MockBadRequest
from .utils import fetch_from_s3, ensure_test_bucket_exists


def make_request(
    filename=None,
    op="sum",
    dtype="int64",
    offset=0,
    size=None,
    order="C",
    shape=[10],
    selection=[[0, 5, 2]],
    missing=None,
):
    """Helper function which by default makes a valid request but can be used to test invalid requests by modifying kwargs"""

    ensure_test_bucket_exists()

    if filename is None:
        filename = s3_client.list_objects_v2(Bucket=BUCKET_NAME)["Contents"][0][
            "Key"
        ]  # Use any valid filename which exists in test bucket

    request_data = {
        "source": S3_SOURCE,
        "bucket": BUCKET_NAME,
        "object": filename,
        "dtype": dtype,
        "offset": offset,
        "size": size,
        "shape": shape,
        "order": order,
        "selection": selection,
        "missing": missing,
    }

    # Remove unset values.
    request_data = {k: v for k, v in request_data.items() if v is not None}

    response = requests.post(
        f"{PROXY_URL}/v1/{op}/", json=request_data, auth=(AWS_ID, AWS_PASSWORD)
    )
    if PROXY_URL is not None:
        print(response.text)  # For debugging

    return response


def test_nonexistent_file(monkeypatch):
    # Generate random file name which should not already exist (unless we're very unlucky)
    invalid_filename = str(uuid.uuid4())
    # Make sure file doesn't actually exist in bucket before making proxy request
    with pytest.raises(FileNotFoundError):
        fetch_from_s3(s3_client, invalid_filename)

    # Make proxy request
    if PROXY_URL is None:
        monkeypatch.setattr(
            requests,
            "post",
            lambda *args, **kwargs: MockBadRequest(),
        )
    response = make_request(filename=invalid_filename)

    # Check the response is sensible
    # Minio returns 404 but radosgw returs 500 so just check response code is something error-like
    assert response.status_code >= 400
    # Check extra stuff if not mocking test result
    if PROXY_URL:
        assert response.headers.get("content-type") == "application/json"
        assert "NoSuchKey" in response.text  # Check for informative error message


def test_invalid_operation(monkeypatch):
    invalid_operation = "this-op-is-not-implented"

    # Make proxy request
    if PROXY_URL is None:
        monkeypatch.setattr(
            requests,
            "post",
            lambda *args, **kwargs: MockBadRequest(),
        )
    response = make_request(op=invalid_operation)

    # Check the response is sensible
    assert response.status_code in (404, 422)
    # Check extra stuff if not mocking test result
    if PROXY_URL:
        assert response.headers.get("content-type") == "application/json"
        assert (
            "operation" in response.text.lower()
        )  # Check for informative error message
        response.json()


def test_invalid_dtype(monkeypatch):
    invalid_dtype = "fake-dtype-64"

    # Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(
            requests,
            "post",
            lambda *args, **kwargs: MockBadRequest(),
        )
    response = make_request(dtype=invalid_dtype)

    # Check the response is sensible
    assert response.status_code in (400, 422)
    # Check extra stuff if not mocking test result
    if PROXY_URL:
        assert response.headers.get("content-type") == "application/json"
        assert "dtype" in response.text.lower()
        response.json()


def test_invalid_offset(monkeypatch):
    invalid_offset = -1

    # Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(
            requests,
            "post",
            lambda *args, **kwargs: MockBadRequest(),
        )
    response = make_request(offset=invalid_offset)

    # Check the response is sensible
    assert response.status_code in (400, 422)
    # Check extra stuff if not mocking test result
    if PROXY_URL:
        assert response.headers.get("content-type") == "application/json"
        assert "offset" in response.text.lower()
        response.json()


def test_invalid_size(monkeypatch):
    invalid_size = -123

    # Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(
            requests,
            "post",
            lambda *args, **kwargs: MockBadRequest(),
        )
    response = make_request(size=invalid_size)

    # Check the response is sensible
    assert response.status_code in (400, 422)
    # Check extra stuff if not mocking test result
    if PROXY_URL:
        assert response.headers.get("content-type") == "application/json"
        assert "size" in response.text.lower()
        response.json()


def test_invalid_shape(monkeypatch):
    invalid_shape = [0]

    # Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(
            requests,
            "post",
            lambda *args, **kwargs: MockBadRequest(),
        )
    response = make_request(shape=invalid_shape)

    # Check the response is sensible
    assert response.status_code in (400, 422)
    # Check extra stuff if not mocking test result
    if PROXY_URL:
        assert response.headers.get("content-type") == "application/json"
        assert "shape" in response.text.lower()  # Check the response is sensible
        response.json()


def test_invalid_selection(monkeypatch):
    invalid_selection = [[10, 100, 1000], [2, 3, 4]]

    # Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(
            requests,
            "post",
            lambda *args, **kwargs: MockBadRequest(),
        )
    response = make_request(selection=invalid_selection)

    # Check the response is sensible
    assert response.status_code == 400
    # Check extra stuff if not mocking test result
    if PROXY_URL:
        assert response.headers.get("content-type") == "application/json"
        assert "selection" in response.text.lower()
        response.json()


def test_shape_without_selection(monkeypatch):
    invalid_shape_and_selection = {"shape": None, "selection": [[10, 1, 0]]}

    # Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(
            requests,
            "post",
            lambda *args, **kwargs: MockBadRequest(),
        )
    response = make_request(**invalid_shape_and_selection)

    # Check the response is sensible
    assert response.status_code == 400
    # Check extra stuff if not mocking test result
    if PROXY_URL:
        assert response.headers.get("content-type") == "application/json"
        assert "shape" in response.text.lower()
        assert "selection" in response.text.lower()
        response.json()


def test_invalid_ordering(monkeypatch):
    invalid_ordering = "nonexistent-ordering"

    # Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(
            requests,
            "post",
            lambda *args, **kwargs: MockBadRequest(),
        )
    response = make_request(order=invalid_ordering)

    # Check the response is sensible
    assert response.status_code == 400
    # Check extra stuff if not mocking test result
    if PROXY_URL:
        assert response.headers.get("content-type") == "application/json"
        assert "order" in response.text.lower()
        response.json()


@pytest.mark.skipif(not MISSING_DATA, reason="Missing data not supported")
@pytest.mark.parametrize(
    "dtype, missing",
    [
        ("int64", "string"),  # Must be a dict
        ("int64", {"invalid_missing": 42}),  # Invalid key
        ("int32", {"missing_value": np.iinfo("int32").min - 1}),  # Less than min
        ("int32", {"missing_value": np.iinfo("int32").max + 1}),  # Greater than max
        ("int32", {"missing_value": -1.0}),  # Float for int
        ("int64", {"missing_value": np.iinfo("int64").min - 1}),  # Less than min
        ("int64", {"missing_value": np.iinfo("int64").max + 1}),  # Greater than max
        ("int64", {"missing_value": -1.0}),  # Float for int
        ("uint32", {"missing_values": [-1]}),  # Negative for unsigned int
        (
            "uint32",
            {"missing_values": [np.iinfo("uint32").max + 1]},
        ),  # Greater than max
        ("uint32", {"valid_min": 1.0}),  # Float for unsigned int
        ("uint64", {"valid_min": [np.iinfo("uint64").max + 1]}),  # Greater than max
        ("uint64", {"valid_min": 1.0}),  # Float for unsigned int
        ("float32", {"valid_max": np.finfo("float32").max * 2}),  # Greater than max
        ("float32", {"valid_max": np.finfo("float32").min * 2}),  # Less than min
        ("float32", {"valid_range": [1.0, 0.0]}),  # min > max
        ("float64", {"valid_range": [1.0, 1.0]}),  # min == max
    ],
)
def test_invalid_missing_data(monkeypatch, dtype, missing):
    # Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(
            requests,
            "post",
            lambda *args, **kwargs: MockBadRequest(),
        )
    response = make_request(dtype=dtype, missing=missing)

    # Check the response is sensible
    assert response.status_code == 400
    # Check extra stuff if not mocking test result
    if PROXY_URL:
        assert response.headers.get("content-type") == "application/json"
        assert "missing" in response.text.lower()
        response.json()
