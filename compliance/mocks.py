
import numpy as np

class MockResponse:

    """ Class used for mocking responses from the active storage proxy implementation in order to facilitate development of this test suite """
    
    def __init__(self, status_code: int, request_data, array_data: np.ndarray, operation_result) -> None:
        selection = request_data['selection']
        # Template response attributes
        self.status_code = status_code
        self.content = operation_result.tobytes()
        self.headers = {
            'content-type': 'application/octet-stream',
            'content-length': str(len(self.content)),
            'x-activestorage-dtype': str(operation_result.dtype),
            'x-activestorage-shape': str(list(operation_result.shape)),
            'x-activestorage-count': str(array_data[tuple(slice(*s) for s in selection)].size if selection else array_data.size),
        }
        return

    # Custom string representation for easier debugging
    def __str__(self) -> str:
        return f'\nMockResponse object:\n Status code: {self.status_code} \n Headers: {self.headers} \n Content: {self.content}'
