
import pytest
import requests
import numpy as np

# Next steps:
# - look at pytest.mark.parameterize decorator for testing functions with multiple inputs
# - create a pytest.ini file and look at 'addopts' pytest option to see if we can use it to set the proxy_url when invoking pytest
# - tests should be written such that if proxy_url is unset then we fall back to request mocking 
# - move MockResponse definitions to separate file 


class MockResponse:
    
    def __init__(self, status_code, content) -> None:
        # Template response attributes
        self.status_code = status_code
        self.content = content
        self.headers = {
            'content-type': 'application/octet-stream',
            'content-length': None,
            'x-activestorage-dtype': None,
            'x-activestorage-shape': None,
        }
        return

    def _update_headers(self, arr: np.array) -> None:
        """ Utility func for setting mock headers automatically given a template np.array """
        if not self.content:
            raise Exception('content attribute must be set before calling this utility function')
        self.headers['content-length'] = len(self.content)
        self.headers['x-activestorage-dtype'] = str(arr.dtype)
        self.headers['x-activestorage-shape'] = list(arr.shape)
        return

    # Custom string representation for easier debugging
    def __str__(self) -> str:
        return f'\nMockResponse object:\n Status code: {self.status_code} \n Headers: {self.headers} \n Content: {self.content}'


def test_sum_int64(monkeypatch):

    # Arrange
    #---------
    request_data = {
        'source': 'https://s3.example.com/path/to/object',
        'dtype': 'int64',
        'offset': 0,
        'size': 100,
        'shape': [20, 5],
        'order': 'C',
        'selection': [[0, 19, 2], [1, 3, 1]],
    }

    slices = tuple(slice(*s) for s in request_data['selection']) #Slices must be a tuple of slice objects for multi-dimensional indexing of numpy arrays
    arr = np.ones(request_data['shape'], dtype=request_data['dtype'])[slices]
    # print(obj.shape) #Check slicing works correctly

    def mock_response(*args, **kwargs):
        r = MockResponse(status_code=200, content=arr.sum().tobytes())
        r._update_headers(arr)
        return r

    monkeypatch.setattr(requests, 'post', mock_response)

    # Act 
    #-----
    response = requests.post('https://s3-proxy.example.com/v1/sum', request_data)
    # print(response)

    # Assert 
    #--------
    assert np.frombuffer(response.content, dtype=response.headers['x-activestorage-dtype']) == arr.sum()