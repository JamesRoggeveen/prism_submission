import equinox as eqx
import jax
from typing import Tuple, Dict, Union
from collections import UserDict

class AbstractData(eqx.Module):
    coords: Tuple[jax.Array, ...]

class CollocationPoints(AbstractData):
    coords: Tuple[jax.Array, ...]

class BoundaryData(AbstractData):
    coords: Tuple[jax.Array, ...]
    normal_vector: Tuple[jax.Array, ...]

class ReferenceData(AbstractData):
    coords: Tuple[jax.Array, ...]
    data: jax.Array

class ProblemData(eqx.Module):
    data: Dict[str, Union[AbstractData, Dict[str, AbstractData]]]

    def __init__(self, **kwargs):
        self.data = {}
        for k, v in kwargs.items():
            if isinstance(v, AbstractData):
                self.data[k] = v
            elif isinstance(v, Dict):
                self.data[k] = {k2: v2 for k2, v2 in v.items() if isinstance(v2, AbstractData)}
            else:
                raise ValueError(f"Value must be an instance of AbstractData or Dict, got {type(v)}")

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value: Union[AbstractData, Dict[str, AbstractData]]):
        if isinstance(value, AbstractData):
            self.data[key] = value
        elif isinstance(value, Dict):
            self.data[key] = {k2: v2 for k2, v2 in value.items() if isinstance(v2, AbstractData)}
        else:
            raise ValueError(f"Value must be an instance of AbstractData or Dict, got {type(value)}")

class SystemConfig(UserDict):
    def __init__(self, **kwargs):
        self.data = kwargs

    def __getattr__(self, name):
        """Called when attribute lookup fails. Falls back to dictionary key lookup."""
        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Called for all attribute assignments (e.g., obj.key = value)."""
        # We need to handle the 'data' attribute assignment from UserDict's __init__
        if name == 'data':
            super().__setattr__(name, value)
        else:
            self.data[name] = value

    def __delattr__(self, name):
        """Called for attribute deletion (e.g., del obj.key)."""
        try:
            del self.data[name]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{name}'")