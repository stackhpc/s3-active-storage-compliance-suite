import abc
from dataclasses import dataclass
import numpy as np
import numpy.ma as ma
import numpy.typing as npt
from typing import Dict, List, Union


class Missing(abc.ABC):
    """Base class for missing data descriptions."""

    """Proportion of elements of the array that should be missing data."""
    p_missing: float = 0.2

    @classmethod
    @abc.abstractmethod
    def create(cls, dtype: np.dtype, operation: str):
        """Create an instance of a subclass with appropriate values for the dtype.

        The created object should affect the result of the operation for the
        specified dtype.
        """

    @abc.abstractmethod
    def to_request_data(self) -> Dict[str, Union[int, float, List]]:
        """Convert the missing data descriptor to the "missing" API request field."""

    @abc.abstractmethod
    def mask(self, arr: npt.NDArray) -> ma.MaskedArray:
        """Apply a mask to an array that matches the missing data."""

    @abc.abstractmethod
    def make_holes(self, arr: npt.NDArray) -> npt.NDArray:
        """Write missing data to random elements of an array."""


def make_holes(
    arr: npt.NDArray, p_missing: float, fill_values: List[Union[int, float]]
) -> npt.NDArray:
    """Write missing data to random elements of an array."""
    rng = np.random.default_rng(10)
    # For N fill values, write p_missing/N of each.
    p_missing = p_missing / len(fill_values)
    for fill_value in fill_values:
        # Generate an array of random bools with the same shape as the data array.
        mask = rng.choice([True, False], arr.shape, p=[p_missing, 1 - p_missing])
        # Mask the data and fill with the fill value.
        arr = ma.array(arr, mask=mask, fill_value=fill_value)
        arr = ma.filled(arr)
    return arr


@dataclass
class MissingValue(Missing):
    """A single missing value."""

    value: Union[int, float]

    @classmethod
    def create(cls, dtype: np.dtype, operation: str) -> Missing:
        by_kind = {
            "i": 999 if operation == "max" else -999,
            "u": 0 if operation == "min" else 999,
            "f": 1e20 if operation == "max" else -1e20,
        }
        return cls(by_kind[dtype.kind])

    def to_request_data(self) -> Dict[str, Union[int, float, List]]:
        return {"missing_value": self.value}

    def mask(self, arr: npt.NDArray) -> ma.MaskedArray:
        return ma.masked_values(arr, self.value)

    def make_holes(self, arr: npt.NDArray) -> npt.NDArray:
        return make_holes(arr, self.p_missing, [self.value])


@dataclass
class MissingValues(Missing):
    """A list of missing values."""

    values: List[Union[int, float]]

    @classmethod
    def create(cls, dtype: np.dtype, operation: str) -> Missing:
        by_kind: Dict[str, List[Union[int, float]]] = {
            "i": [-999, 1000],
            "u": [0, 999],
            "f": [-1e20, 1e20],
        }
        return cls(by_kind[dtype.kind])

    def to_request_data(self) -> Dict[str, Union[int, float, List]]:
        return {"missing_values": self.values}

    def mask(self, arr: npt.NDArray) -> ma.MaskedArray:
        return ma.masked_where(np.isin(arr, self.values), arr)

    def make_holes(self, arr: npt.NDArray) -> npt.NDArray:
        return make_holes(arr, self.p_missing, self.values)


@dataclass
class ValidMax(Missing):
    """A valid maximum value."""

    max: Union[int, float]

    @classmethod
    def create(cls, dtype: np.dtype, operation: str) -> Missing:
        by_kind = {
            "i": 8,
            "u": 9,
            "f": 10.0,
        }
        return cls(by_kind[dtype.kind])

    def to_request_data(self) -> Dict[str, Union[int, float, List]]:
        return {"valid_max": self.max}

    def mask(self, arr: npt.NDArray) -> ma.MaskedArray:
        return ma.masked_greater(arr, self.max)

    def make_holes(self, arr: npt.NDArray) -> npt.NDArray:
        return make_holes(arr, self.p_missing, [self.max + 1])


@dataclass
class ValidMin(Missing):
    """A valid minimum value."""

    min: Union[int, float]

    @classmethod
    def create(cls, dtype: np.dtype, operation: str) -> Missing:
        by_kind = {
            "i": -1,
            "u": 2,
            "f": 0.5,
        }
        return cls(by_kind[dtype.kind])

    def to_request_data(self) -> Dict[str, Union[int, float, List]]:
        return {"valid_min": self.min}

    def mask(self, arr: npt.NDArray) -> ma.MaskedArray:
        return ma.masked_less(arr, self.min)

    def make_holes(self, arr: npt.NDArray) -> npt.NDArray:
        return make_holes(arr, self.p_missing, [self.min - 1])


@dataclass
class ValidRange(Missing):
    """A valid range of values."""

    min: Union[int, float]
    max: Union[int, float]

    @classmethod
    def create(cls, dtype: np.dtype, operation: str) -> Missing:
        by_kind = {
            "i": (2, 10),
            "u": (1, 9),
            "f": (0.5, 9.9),
        }
        return cls(*by_kind[dtype.kind])

    def to_request_data(self) -> Dict[str, Union[int, float, List]]:
        return {"valid_range": [self.min, self.max]}

    def mask(self, arr: npt.NDArray) -> ma.MaskedArray:
        return ma.masked_outside(arr, self.min, self.max)

    def make_holes(self, arr: npt.NDArray) -> npt.NDArray:
        return make_holes(arr, self.p_missing, [self.min - 1, self.max + 1])
