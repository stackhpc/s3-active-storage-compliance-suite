import json
import numpy as np
import numpy.ma as ma
from typing import Dict


class BaseMockResponse(object):
    """Class used for mocking responses from the active storage proxy implementation in order to facilitate development of this test suite"""

    def __init__(
        self,
        status_code: int,
        content: bytes,
        headers: Dict[str, str],
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.content = content

    # Custom string representation for easier debugging
    def __str__(self) -> str:
        return f"\nMockResponse object:\n Status code: {self.status_code} \n Headers: {self.headers} \n Content: {self.content!r}"


class MockResponse(BaseMockResponse):
    def __init__(
        self, status_code: int, array_data: np.ndarray, operation_result, order="C"
    ) -> None:
        content = operation_result.tobytes(order=order)
        headers = {
            "content-type": "application/octet-stream",
            "content-length": str(len(self.content)),
            "x-activestorage-dtype": str(operation_result.dtype),
            "x-activestorage-shape": str(list(operation_result.shape)),
            "x-activestorage-count": str(ma.count(array_data)),
        }
        super(MockResponse, self).__init__(status_code, content, headers)


class MockErrorResponse(BaseMockResponse):
    def __init__(self, status_code: int) -> None:
        content = json.dumps({"errors": []}).encode()
        headers = {
            "content-type": "application/json",
            "content-length": str(len(self.content)),
        }
        super(MockErrorResponse, self).__init__(status_code, content, headers)


class MockBadRequest(MockErrorResponse):
    """400 Bad Request"""

    def __init__(self):
        super(MockBadRequest, self).__init__(400)
