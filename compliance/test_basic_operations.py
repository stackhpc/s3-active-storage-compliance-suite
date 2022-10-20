
from os import PRIO_PGRP
from urllib import request
import pytest
import requests
import numpy as np
from .config import s3_client, S3_SOURCE, PROXY_URL, BUCKET_NAME, ALLOWED_DTYPES, OPERATION_FUNCS
from .utils import upload_to_s3, fetch_from_s3
from .mocks import MockResponse


def generate_test_data(filename, operation: str, dtype: str, shape: list[int], selection: list[list[int]]):

    """ Creates some test data and uploads it to the configured S3 source for later requests through the active proxy """

    np.random.seed(10) #Make sure randomized arrays are reproducible

    #Generate some test data (multiply random array by 100 so that int dtypes don't all round down to zeros)
    data = (100*np.random.rand(*shape)).astype(dtype)
    #Add data to s3 bucket so that proxy can use it
    upload_to_s3(s3_client, data, filename)
    #Perform main operation (after slicing if necessary)
    if selection is not None:
        #Create pythonic slices object (must be a tuple of slice objects for multi-dimensional indexing of numpy arrays)
        slices = tuple(slice(*s) for s in selection)
        operation_result = OPERATION_FUNCS[operation](data[slices])
    else:
        operation_result = OPERATION_FUNCS[operation](data)

    return operation_result


#Stacking parametrization decorators tells pytest to check every possible combination of parameters
@pytest.mark.parametrize('operation', OPERATION_FUNCS.keys())
@pytest.mark.parametrize('dtype', ALLOWED_DTYPES)
@pytest.mark.parametrize('shape, selection', [([10], None), ([100], [[10, 50, 4]]), ([20, 5], [[0, 19, 2], [1, 3, 1]])])
def test_basic_operation(monkeypatch, operation, dtype, shape, selection):

    """ Test basic functionality of reduction operations on various types of input data """

    filename = f"test--dtype-{dtype}--shape-{shape}--selection-{selection}.bin"
    operation_result = generate_test_data(filename, operation, dtype, shape, selection)

    request_data = {
        'source': f'{S3_SOURCE}/{BUCKET_NAME}/{filename}',
        'dtype': dtype,
        'offset': 0,
        'size': 100,
        'shape': shape,
        'order': 'C',
        'selection': selection,
    }

    #Mock proxy responses if url not set
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=200, operation_result=operation_result))

    # Fetch response from proxy
    proxy_response = requests.post(f'{PROXY_URL}/v1/{operation}', request_data)

    # Compare to expected result and make sure response headers are sensible
    assert proxy_response.content == operation_result.tobytes()
    assert proxy_response.headers['x-activestorage-dtype'] == (request_data['dtype'] if operation != 'count' else 'int64') #Count should always return an int64 according to project spec
    assert proxy_response.headers['x-activestorage-shape'] == list(operation_result.shape)
    assert proxy_response.headers['content-length'] == operation_result.size