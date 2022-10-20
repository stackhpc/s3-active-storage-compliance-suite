
import uuid
import requests
import pytest
import numpy as np

from .config import s3_client, S3_SOURCE, BUCKET_NAME, PROXY_URL, OPERATION_FUNCS
from .utils import fetch_from_s3
from .mocks import MockResponse



def test_nonexistent_file(monkeypatch):

    #Generate random file name which should not already exist (unless we're very unlucky)
    filename = str(uuid.uuid4())
    url = f'{PROXY_URL}/v1/{filename}'
    #Make sure file doesn't actually exist in bucket before making proxy request
    with pytest.raises(FileNotFoundError):
        fetch_from_s3(s3_client, filename)
       
    #Make proxy request
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=404, operation_result=np.array([]))) #Is this a sensible mock response?
    response = requests.post(url)

    #Check the response is sensible
    assert response.status_code == 404
    assert response.headers['content-length'] == 0
    # assert <something-about-informative-error-message-in-response-body?>

    

def test_invalid_operation(monkeypatch):

    invalid_operation = 'this-op-is-not-implented'
    #Filename shouldn't matter but better to request a file that actually exists to avoid 
    # proxy implementation details of whether file existence is validated first or not
    # - use the first filename returned in the list of all objects within the bucket
    filename = s3_client.list_objects_v2(Bucket=BUCKET_NAME)['Contents'][0]['Key']
    request_data = {
        'source': f'{S3_SOURCE}/{BUCKET_NAME}/{filename}',
        'dtype': 'int64',
        'offset': 0,
        'size': 10,
        'shape': [10],
        'order': 'C',
        'selection': [None],
    }

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    response = requests.post(f'{PROXY_URL}/v1/{invalid_operation}', request_data)

    #Check the response is sensible
    assert response.status_code == 400
    assert response.headers['content-length'] == 0
    # assert <something-about-informative-error-message-in-response-body?>
 


def test_invalid_dtype(monkeypatch):

    invalid_dtype = 'fake-dtype-64'
    #Filename shouldn't matter but better to request a file that actually exists to avoid 
    # proxy implementation details of whether file existence is validated before anything 
    # else - use the first filename returned in the list of all objects within the bucket
    filename = s3_client.list_objects_v2(Bucket=BUCKET_NAME)['Contents'][0]['Key']

    request_data = {
        'source': f'{S3_SOURCE}/{BUCKET_NAME}/{filename}',
        'dtype': invalid_dtype,
        'offset': 0,
        'size': 10,
        'shape': [10],
        'order': 'C',
        'selection': [None],
    }

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    op = next(iter(OPERATION_FUNCS.keys())) #Operation doesn't matter but be sure to use a valid one for the same reasons as filename
    response = requests.post(f'{PROXY_URL}/v1/{op}', request_data) 

    #Check the response is sensible
    assert response.status_code == 400
    assert response.headers['content-length'] == 0
    # assert <something-about-informative-error-message-in-response-body?>


def test_invalid_offset(monkeypatch):

    invalid_offset = 'abc'
    #Filename shouldn't matter but better to request a file that actually exists to avoid 
    # proxy implementation details of whether file existence is validated before anything 
    # else - use the first filename returned in the list of all objects within the bucket
    filename = s3_client.list_objects_v2(Bucket=BUCKET_NAME)['Contents'][0]['Key']

    request_data = {
        'source': f'{S3_SOURCE}/{BUCKET_NAME}/{filename}',
        'dtype': 'float64',
        'offset': invalid_offset,
        'size': 10,
        'shape': [10],
        'order': 'C',
        'selection': [None],
    }

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    op = next(iter(OPERATION_FUNCS.keys())) #Operation doesn't matter but be sure to use a valid one for the same reasons as filename
    response = requests.post(f'{PROXY_URL}/v1/{op}', request_data)

    #Check the response is sensible
    assert response.status_code == 400
    assert response.headers['content-length'] == 0
    # assert <something-about-informative-error-message-in-response-body?>



def test_invalid_size(monkeypatch):

    invalid_size = -123
    #Filename shouldn't matter but better to request a file that actually exists to avoid 
    # proxy implementation details of whether file existence is validated before anything 
    # else - use the first filename returned in the list of all objects within the bucket
    filename = s3_client.list_objects_v2(Bucket=BUCKET_NAME)['Contents'][0]['Key']

    request_data = {
        'source': f'{S3_SOURCE}/{BUCKET_NAME}/{filename}',
        'dtype': 'float64',
        'offset': 0,
        'size': invalid_size,
        'shape': [10],
        'order': 'C',
        'selection': [None],
    }

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    op = next(iter(OPERATION_FUNCS.keys())) #Operation doesn't matter but be sure to use a valid one for the same reasons as filename
    response = requests.post(f'{PROXY_URL}/v1/{op}', request_data)

    #Check the response is sensible
    assert response.status_code == 400
    assert response.headers['content-length'] == 0
    # assert <something-about-informative-error-message-in-response-body?>



def test_invalid_shape(monkeypatch):

    invalid_shape = list(range(100))
    #Filename shouldn't matter but better to request a file that actually exists to avoid 
    # proxy implementation details of whether file existence is validated before anything 
    # else - use the first filename returned in the list of all objects within the bucket
    filename = s3_client.list_objects_v2(Bucket=BUCKET_NAME)['Contents'][0]['Key']

    request_data = {
        'source': f'{S3_SOURCE}/{BUCKET_NAME}/{filename}',
        'dtype': 'float64',
        'offset': 0,
        'size': 100,
        'shape': invalid_shape,
        'order': 'C',
        'selection': [None],
    }

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    op = next(iter(OPERATION_FUNCS.keys())) #Operation doesn't matter but be sure to use a valid one for the same reasons as filename
    response = requests.post(f'{PROXY_URL}/v1/{op}', request_data)

    #Check the response is sensible
    assert response.status_code == 400
    assert response.headers['content-length'] == 0
    # assert <something-about-informative-error-message-in-response-body?>



def test_invalid_selection(monkeypatch):

    invalid_selection = [[10, 100, 1000], [2, 3, 4]]
    #Filename shouldn't matter but better to request a file that actually exists to avoid 
    # proxy implementation details of whether file existence is validated before anything 
    # else - use the first filename returned in the list of all objects within the bucket
    filename = s3_client.list_objects_v2(Bucket=BUCKET_NAME)['Contents'][0]['Key']

    request_data = {
        'source': f'{S3_SOURCE}/{BUCKET_NAME}/{filename}',
        'dtype': 'float64',
        'offset': 0,
        'size': 100,
        'shape': [10],
        'order': 'C',
        'selection': invalid_selection,
    }

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    op = next(iter(OPERATION_FUNCS.keys())) #Operation doesn't matter but be sure to use a valid one for the same reasons as filename
    response = requests.post(f'{PROXY_URL}/v1/{op}', request_data)

    #Check the response is sensible
    assert response.status_code == 400
    assert response.headers['content-length'] == 0
    # assert <something-about-informative-error-message-in-response-body?>



def test_invalid_ordering(monkeypatch):

    invalid_ordering = 'nonexistent-ordering'
    #Filename shouldn't matter but better to request a file that actually exists to avoid 
    # proxy implementation details of whether file existence is validated before anything 
    # else - use the first filename returned in the list of all objects within the bucket
    filename = s3_client.list_objects_v2(Bucket=BUCKET_NAME)['Contents'][0]['Key']

    request_data = {
        'source': f'{S3_SOURCE}/{BUCKET_NAME}/{filename}',
        'dtype': 'float64',
        'offset': 0,
        'size': 100,
        'shape': [10],
        'order': invalid_ordering,
        'selection': [None],
    }

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    op = next(iter(OPERATION_FUNCS.keys())) #Operation doesn't matter but be sure to use a valid one for the same reasons as filename
    response = requests.post(f'{PROXY_URL}/v1/{op}', request_data)

    #Check the response is sensible
    assert response.status_code == 400
    assert response.headers['content-length'] == 0
    # assert <something-about-informative-error-message-in-response-body?>
