

class MockResponse:

    """ Class used for mocking responses from the active storage proxy implementation in order to facilitate development of this test suite """
    
    def __init__(self, status_code: int, operation_result) -> None:
        # Template response attributes
        self.status_code = status_code
        self.content = operation_result.tobytes()
        self.headers = {
            'content-type': 'application/octet-stream',
            'content-length': operation_result.size,
            'x-activestorage-dtype': str(operation_result.dtype),
            'x-activestorage-shape': list(operation_result.shape),
        }
        return

    # Custom string representation for easier debugging
    def __str__(self) -> str:
        return f'\nMockResponse object:\n Status code: {self.status_code} \n Headers: {self.headers} \n Content: {self.content}'
