
import uuid
import requests
import pytest
import numpy as np

from .config import s3_client, S3_SOURCE, BUCKET_NAME, PROXY_URL, AWS_ID, AWS_PASSWORD
from .utils import fetch_from_s3, ensure_test_bucket_exists
from .mocks import MockResponse


def make_request(
    filename = None,
    op = 'sum',
    dtype = 'int64',
    offset = 0,
    size = None,
    order = 'C',
    shape = [10],
    selection = [[0, 5, 2]]   
):

    """ Helper function which by default makes a valid request but can be used to test invalid requests by modifying kwargs """

    ensure_test_bucket_exists()

    if filename is None:
        filename = s3_client.list_objects_v2(Bucket=BUCKET_NAME)['Contents'][0]['Key'] #Use any valid filename which exists in test bucket

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

    response = requests.post(f'{PROXY_URL}/v1/{op}/', json=request_data, auth=(AWS_ID, AWS_PASSWORD))
    print(response.text) #For debugging

    return response



def test_nonexistent_file(monkeypatch):

    #Generate random file name which should not already exist (unless we're very unlucky)
    invalid_filename = str(uuid.uuid4())
    #Make sure file doesn't actually exist in bucket before making proxy request
    with pytest.raises(FileNotFoundError):
        fetch_from_s3(s3_client, invalid_filename)
    
    #Make proxy request
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=404, operation_result=np.array([]))) #Is this a sensible mock response?
    response = make_request(filename=invalid_filename)

    #Check the response is sensible
    # Minio returns 404 but radosgw returs 500 so just check response code is something error-like
    assert response.status_code >= 400
    assert response.headers['content-type'] == 'application/json'
    # assert <something-about-informative-error-message-in-response-body?>

    

def test_invalid_operation(monkeypatch):

    invalid_operation = 'this-op-is-not-implented'

    #Make proxy request
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=404, operation_result=np.array([]))) #Is this a sensible mock response?
    response = make_request(op=invalid_operation)

    #Check the response is sensible
    assert response.status_code == 422
    assert response.headers['content-type'] == 'application/json'
    assert 'operation' in response.text.lower() #Check for informative error message
    

def test_invalid_dtype(monkeypatch):

    invalid_dtype = 'fake-dtype-64'

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    response = make_request(dtype=invalid_dtype)

    #Check the response is sensible
    assert response.status_code == 422
    assert 'not' in response.text.lower()
    assert 'valid' in response.text.lower()
    assert 'dtype' in response.text.lower()



def test_invalid_offset(monkeypatch):

    invalid_offset = -1

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    response = make_request(offset=invalid_offset)

    #Check the response is sensible
    assert response.status_code == 422
    assert 'offset' in response.text.lower()


def test_invalid_size(monkeypatch):

    invalid_size = -123

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    response = make_request(size=invalid_size)

    #Check the response is sensible
    assert response.status_code == 422
    assert 'size' in response.text.lower()



def test_invalid_shape(monkeypatch):

    invalid_shape = [0]

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    response = make_request(shape=invalid_shape)

    #Check the response is sensible
    assert response.status_code == 422
    assert 'shape' in response.text.lower()    #Check the response is sensible



def test_invalid_selection(monkeypatch):

    invalid_selection = [[10, 100, 1000], [2, 3, 4]]

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    response = make_request(selection=invalid_selection)

    #Check the response is sensible
    assert response.status_code == 400
    assert 'selection' in response.text.lower()


def test_shape_without_selection(monkeypatch):
    
    invalid_shape_and_selection = {'shape': None, 'selection': [[10, 1, 0]]}

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    response = make_request(**invalid_shape_and_selection)

    #Check the response is sensible
    assert response.status_code == 400
    assert 'shape' in response.text.lower()
    assert 'selection' in response.text.lower()



def test_invalid_ordering(monkeypatch):

    invalid_ordering = 'nonexistent-ordering'

    #Make proxy request (mocking response if needed)
    if PROXY_URL is None:
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse(status_code=400, operation_result=np.array([])))
    response = make_request(order=invalid_ordering)

    #Check the response is sensible
    assert response.status_code == 400
    assert 'order' in response.text.lower()