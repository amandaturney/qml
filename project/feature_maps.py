from math import pi
import numpy as np
from qiskit.circuit.library import PauliFeatureMap
from functools import partial

from typing import Callable


encoding_fns = {
    'encoding_fn_1': lambda x1, x2: pi*x1*x2,
    'encoding_fn_2': lambda x1, x2: (pi/2)*(1-x1)*(1-x2),
    'encoding_fn_3': lambda x1, x2: np.exp(((x1-x2)*(x1-x2))/(8/np.log(pi))),
    'encoding_fn_4': lambda x1, x2: (pi)/(3*np.cos(x1)*np.cos(x2)),
    'encoding_fn_5': lambda x1, x2: pi*np.cos(x1)*np.cos(x2)
}



def param_circuit(encoding_fn: Callable) -> PauliFeatureMap:
    """
    Returns a PauliFeatureMap parameterized quantum circuit that encodes
    the given encoding function.

    Args:
        encoding_fn (Callable): encoding function of signiture (x1, x2) -> x_1,2

    Returns:
        PauliFeatureMap: feature map
    """
    # First define a custom data map based off a given encoding function
    def custom_data_map_func(func, x):
        coeff = x[0] if len(x) == 1 else func(x[0], x[1])
        return coeff

    # Then use the partial function to freeze the encoding function
    data_map_func = partial(custom_data_map_func, encoding_fn)
    return PauliFeatureMap(feature_dimension=2, reps=2, entanglement='linear', alpha=1.0, data_map_func=data_map_func)