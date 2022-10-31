
import pytest
import requests
import numpy as np

from typing import Union
from .config import s3_client, S3_SOURCE, PROXY_URL, BUCKET_NAME, ALLOWED_DTYPES, OPERATION_FUNCS, AWS_ID, AWS_PASSWORD
from .utils import upload_to_s3, ensure_test_bucket_exists
from .mocks import MockResponse


def generate_test_data(filename, operation: str, dtype: str, shape: list[int], selection: list[list[int]], offset: Union[int, None]):

    """ Creates some test data and uploads it to the configured S3 source for later requests through the active proxy """

    np.random.seed(10) #Make sure randomized arrays are reproducible

    #Generate some test data (multiply random array by 10 so that int dtypes don't all round down to zeros)
    data = (10*np.random.rand(*shape)).astype(dtype)
    #Add data to s3 bucket so that proxy can use it
    ensure_test_bucket_exists()
    upload_to_s3(s3_client, data, filename)
    #Perform any required offsetting and slicing
    if offset:
        #Use numpy's builtin offset functionality
        print(data)
        print('\nSlicing')
        data = np.frombuffer(data.tobytes(), dtype=dtype, offset=offset)
        print(data)
    if selection:
        #Create pythonic slices object (must be a tuple of slice objects for multi-dimensional indexing of numpy arrays)
        slices = tuple(slice(*s) for s in selection)
        data = data[slices]
 
    #Perform main operation
    operation_result = OPERATION_FUNCS[operation](data)

    return operation_result


#Stacking parametrization decorators tells pytest to check every possible combination of parameters
@pytest.mark.parametrize('operation', OPERATION_FUNCS.keys())
@pytest.mark.parametrize('dtype', ALLOWED_DTYPES)
@pytest.mark.parametrize('shape, selection', [([10], None), ([100], [[10, 50, 4]]), ([20, 5], [[0, 19, 2], [1, 3, 1]])])
@pytest.mark.parametrize('offset', [None])
def test_basic_operation(monkeypatch, operation, dtype, shape, selection, offset):

    """ Test basic functionality of reduction operations on various types of input data """

    filename = f"test--dtype-{dtype}--shape-{shape}--selection-{selection}.bin"
    operation_result = generate_test_data(filename, operation, dtype, shape, selection, offset)

    request_data = {
        'source': S3_SOURCE,
        'bucket': BUCKET_NAME,
        'object': filename,
        'dtype': dtype,
        'offset': offset,
        'size': None,
        'shape': shape,
        'order': 'C',
        'selection': selection,
    }

    #Mock proxy responses if url not set
    # print(PROXY_URL)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=200, operation_result=operation_result))

    # Fetch response from proxy
    proxy_response = requests.post(f'{PROXY_URL}/v1/{operation}/', json=request_data, auth=(AWS_ID, AWS_PASSWORD))
    # print(proxy_response.text)
    assert proxy_response.status_code == 200

    proxy_result = np.frombuffer(proxy_response.content, dtype=proxy_response.headers['x-activestorage-dtype'])

    # Compare to expected result and make sure response headers are sensible - all comparisons should be done as strings
    print('\nProxy result:', proxy_result, '\nExpected result:', operation_result) #For debugging
    assert proxy_response.headers['x-activestorage-dtype'] == (request_data['dtype'] if operation != 'count' else 'int64') #Count should always return an int64 according to project spec
    assert proxy_response.headers['x-activestorage-shape'] == str(list(operation_result.shape)) if operation != 'count' else '[1]'
    assert proxy_response.content == operation_result.tobytes()
    assert proxy_response.headers['content-length'] == str(len(operation_result.tobytes()))