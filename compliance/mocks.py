import numpy as np
import numpy.ma as ma


class MockResponse:

    """Class used for mocking responses from the active storage proxy implementation in order to facilitate development of this test suite"""

    def __init__(
        self, status_code: int, array_data: np.ndarray, operation_result, order="C"
    ) -> None:
        # Template response attributes
        self.status_code = status_code
        self.content = operation_result.tobytes(order=order)
        self.headers = {
            "content-type": "application/octet-stream",
            "content-length": str(len(self.content)),
            "x-activestorage-dtype": str(operation_result.dtype),
            "x-activestorage-shape": str(list(operation_result.shape)),
            "x-activestorage-count": str(ma.count(array_data)),
        }
        return

    # Custom string representation for easier debugging
    def __str__(self) -> str:
        return f"\nMockResponse object:\n Status code: {self.status_code} \n Headers: {self.headers} \n Content: {self.content}"
