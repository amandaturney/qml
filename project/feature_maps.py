from math import pi
import numpy as np
from functools import partial
from qiskit.circuit.library import PauliFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from typing import Callable, Optional


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


def calculate_kernel(
        feature_map: PauliFeatureMap,
        train_X: np.ndarray,
        test_X: Optional[np.ndarray] = None,
        shots: Optional[int] = None,
        sampler_seed: Optional[int] = None,
        backend: str = 'qasm_simulator'
    ) -> np.ndarray:
    """
    Simulates or calculates the kernel matrix given the feature map and training and/or testing data.
    If only training data is provided, it computes the inner product of the training data with itself;
    otherwise it computes the fidelities between the training and testing data. If `shots` is None, 
    then it calculates the probabilities. Otherwise, it samples the number of `shots` from the multinomial
    distribution to compute the kernel matrix.

    Args:
        feature_map (PauliFeatureMap): PauliFeatureMap / Quantum circuit of the feature map
        train_X (np.ndarray): training data array, M x N, where M is the number of samples
            and N is the number of features
        test_X (Optional[np.ndarray], optional): test data array, M x N, where M is the
            number of test samples and N is the number of features. Defaults to None.
        shots (Optional[int], optional): Number of shots for the sampler. If none is provided,
            then it computes the probabilities. Defaults to None.
        sampler_seed (Optional[int], optional): Random seed for the sampler; is ignored if
            `shots` is None. Defaults to None.
        backend (str, optional): Quantum backend to use. Defaults to 'qasm_simulator'.

    Returns:
        np.ndarray: kernel matrix
    """
    sampler = Sampler(options=dict(shots=shots, seed=sampler_seed, backend=backend))
    fidelity = ComputeUncompute(sampler=sampler)
    adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

    if test_X is None:
        kernel = adhoc_kernel.evaluate(x_vec=train_X)
    else:
        # Note that when passing both training & testing data, your TESTING data now becomes
        # the new x_vec argument and your training data is passed as y_vec. If reversed, your
        # kernel matrix will be transposed
        kernel = adhoc_kernel.evaluate(x_vec=test_X, y_vec=train_X)
    
    return kernel