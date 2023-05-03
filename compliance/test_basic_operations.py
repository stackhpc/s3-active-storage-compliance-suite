
import json
import pytest
import requests
import numpy as np

from typing import Union
from .config import s3_client, S3_SOURCE, PROXY_URL, BUCKET_NAME, ALLOWED_DTYPES, OPERATION_FUNCS, AWS_ID, AWS_PASSWORD
from .utils import upload_to_s3, ensure_test_bucket_exists
from .mocks import MockResponse


def generate_test_data(
        filename, 
        operation: str, 
        dtype: str,
        shape: list[int], 
        selection: list[list[int]], 
        offset: Union[int, None], 
        size: Union[int, None], 
        order: str
    ):

    """ 
    Creates some test data and uploads it to the configured
    S3 source for later requests through the active proxy 
    """

    np.random.seed(10) #Make sure randomized arrays are reproducible

    #Generate some test data 
    #This is the raw data that will be uploaded to S3, and is currently a 1D array.
    # (multiply random array by 10 so that int dtypes don't all round down to zeros)
    data = (10*np.random.rand(100)).astype(dtype) 

    #Add data to s3 bucket so that proxy can use it
    ensure_test_bucket_exists()
    upload_to_s3(s3_client, data, filename)

    #Perform any required offsetting and other chunk manipulations
    offset = offset or 0
    count = -1 # value = -1 tells numpy to read whole buffer
    if size:
        count = size // np.dtype(dtype).itemsize
    data_bytes = data.tobytes()
    #Create a 1D array from the required byte range.
    data = np.frombuffer(data_bytes, dtype, offset=offset, count=count)

    #Reshape the array to apply the shape (if specified) and C/F order.
    data = data.reshape(*(shape or data.shape), order=order)

    #Create pythonic slices object 
    # (must be a tuple of slice objects for multi-dimensional indexing of numpy arrays)
    if selection:
        slices = tuple(slice(*s) for s in selection)
        data = data[slices]
 
    #Perform main operation
    operation_result = OPERATION_FUNCS[operation](data)

    return operation_result


#Stacking parametrization decorators tells pytest to check every possible combination of parameters
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('shape, selection', [(None, None), ([5, 5, 4], None), ([100], [[-10, -50, -4]]), ([20, 5], [[0, 19, 2], [1, 3, 1]])])
@pytest.mark.parametrize('dtype', ALLOWED_DTYPES)
@pytest.mark.parametrize('operation', OPERATION_FUNCS.keys())
def test_basic_operation(monkeypatch, operation, dtype, shape, selection, order, offset=None, size=None):

    """ Test basic functionality of reduction operations on various types of input data """

    filename = f"test--dtype-{dtype}--shape-{shape}--selection-{selection}.bin"
    operation_result = generate_test_data(filename, operation, dtype, shape, selection, offset, size, order)

    request_data = {
        'source': S3_SOURCE,
        'bucket': BUCKET_NAME,
        'object': filename,
        'dtype': dtype,
        'offset': offset,
        'size': size,
        'shape': shape,
        'order': order,
        'selection': selection,
    }

    #Mock proxy responses if url not set
    if PROXY_URL is None:
        monkeypatch.setattr(
            requests, 
            'post', 
            lambda *args, **kwargs: MockResponse(status_code=200, operation_result=operation_result)
        )

    # Fetch response from proxy
    proxy_response = requests.post(f'{PROXY_URL}/v1/{operation}/', json=request_data, auth=(AWS_ID, AWS_PASSWORD))

    #For debugging failed tests
    if proxy_response.status_code != 200:
        print(proxy_response.text)

    assert proxy_response.status_code == 200

    proxy_result = np.frombuffer(proxy_response.content, dtype=proxy_response.headers['x-activestorage-dtype'])

    # Compare to expected result and make sure response headers are sensible - all comparisons should be done as strings
    print('\nProxy result:', proxy_result, '\nExpected result:', operation_result) #For debugging
    assert proxy_response.headers['x-activestorage-dtype'] == (request_data['dtype'] if operation != 'count' else 'int64')
    expected_shape = list(operation_result.shape)
    proxy_shape = json.loads(proxy_response.headers['x-activestorage-shape'])
    assert proxy_shape == expected_shape
    proxy_result = proxy_result.reshape(proxy_shape, order=order)
    assert np.allclose(proxy_result, operation_result), f"actual:\n{proxy_result}\n!=\nexpected:\n{operation_result}"
    assert proxy_response.headers['content-length'] == str(len(operation_result.tobytes()))



#Separate out these tests since valid offset & size values depend on other parameters so combinatorial param approach is too complicated
param_combos = [
    (
        'int64', #dtype
        [20],    #shape
        None,    #selection
        8*10,    #offset
        8*20,    #size - must equal product of shape * dtype size in bytes
    ),
    (
        'float32',
        [10, 3],
        [[0, 10, 2], [0, 3, 1]],
        4*2,
        4*30,
    ),
    (
        'uint32',
        [10, 2, 4],
        [[0, 10, 2], [0, 2, 1], [0, 3, 1]],
        4*5,
        4*80,
    ),
        (
        'int32',
        [10, 2, 4],
        [[0, 10, 2], [0, 2, 1], [0, 4, 3]],
        4*20,
        None,
    ),
]
@pytest.mark.parametrize('dtype, shape, selection, offset, size', param_combos)
@pytest.mark.parametrize('operation', OPERATION_FUNCS.keys())
@pytest.mark.parametrize('order', ['C', 'F'])
def test_offset_and_size(monkeypatch, operation, dtype, shape, selection, order, offset, size):
    #We can still hook into previous test func though to avoid repeated code 
    #(maybe there's a more pytest-y way to do this?)
    test_basic_operation(monkeypatch, operation, dtype, shape, selection, order, offset, size)
