import numpy as np
from numpy import sin, cos
from collections import OrderedDict
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector
from qiskit.quantum_info.operators.symplectic.pauli_utils import pauli_basis
from qiskit.circuit.library import PauliFeatureMap
from typing import Callable, Dict, List, Optional, Tuple, Union


coefficients = {
        'II': lambda x1, x2, x12: np.ones_like(x12)*1/4,
        'XI': lambda x1, x2, x12: (sin(x2)*(sin(x1)*sin(x12)**2 + sin(x2)*cos(x12)**2 + cos(x1)*cos(x2)*sin(x12)))/4,
        'YI': lambda x1, x2, x12: (-sin(x1)*cos(x2)*sin(x12)**2 - sin(x2)*cos(x2)*cos(x12)**2 + cos(x1)*sin(x12)*sin(x2)**2)/4,
        'ZI': lambda x1, x2, x12: cos(x2)*cos(x12)/4,
        'IX': lambda x1, x2, x12: (sin(x1)*(sin(x2)*sin(x12)**2 + sin(x1)*cos(x12)**2 + cos(x2)*cos(x1)*sin(x12)))/4,
        'XX': lambda x1, x2, x12: ((sin(x1)**2)*(sin(x2)**2) + sin(x12)*cos(x1)*cos(x2)*(sin(x1)+sin(x2)))/4,
        'YX': lambda x1, x2, x12: ((-sin(x1)**2)*sin(x2)*cos(x2) + sin(x12)*cos(x1)*(sin(x1)*sin(x2) - cos(x2)**2))/4,
        'ZX': lambda x1, x2, x12: (cos(x12)*(-sin(x2)*cos(x1)*sin(x12) + cos(x2)*sin(x1)**2 + sin(x1)*cos(x1)*sin(x12)))/4,
        'IY': lambda x1, x2, x12: (-sin(x2)*cos(x1)*sin(x12)**2 - sin(x1)*cos(x1)*cos(x12)**2 + cos(x2)*sin(x12)*sin(x1)**2)/4,
        'XY': lambda x1, x2, x12: ((-sin(x2)**2)*sin(x1)*cos(x1) + sin(x12)*cos(x2)*(sin(x1)*sin(x2) - cos(x1)**2))/4,
        'YY': lambda x1, x2, x12: (sin(x1)*cos(x1)*sin(x2)*cos(x2) - sin(x12)*((cos(x2)**2)*sin(x1) + sin(x2)*cos(x1)**2))/4,
        'ZY': lambda x1, x2, x12: (sin(x1)*(-sin(x2)*sin(x12)*cos(x12) - cos(x1)*cos(x2)*cos(x12) + sin(x1)*cos(x12)*sin(x12)))/4,
        'IZ': lambda x1, x2, x12: cos(x1)*cos(x12)/4,
        'XZ': lambda x1, x2, x12: (cos(x12)*(-sin(x1)*cos(x2)*sin(x12) + cos(x1)*sin(x2)**2 + sin(x2)*cos(x2)*sin(x12)))/4,
        'YZ': lambda x1, x2, x12: (sin(x2)*(-sin(x1)*sin(x12)*cos(x12) - cos(x2)*cos(x1)*cos(x12) + sin(x2)*cos(x12)*sin(x12)))/4,
        'ZZ': lambda x1, x2, x12: cos(x1)*cos(x2)/4
    }

# Get basis matrices, and put in same order as coefficients above
pauli_basis_list = pauli_basis(2)
basis_pauli_matrices = dict(zip(pauli_basis_list.label_iter(), pauli_basis_list.to_matrix()))
basis_matrices_2qubit = [basis_pauli_matrices[axis] for axis in coefficients.keys()]


def convert_to_pauli_decomposition_old(
        ds: np.ndarray, feature_map: PauliFeatureMap
    ) -> Tuple[np.ndarray, List[str]]:
    """
    Given a dataset of shape (M x N) where M is the number of samples and
    N is the number of features, this function returns a 2D array of the
    quantum representation of the dataset given a feature map. This
    representation first computes the state vector of the data-encoded
    quantum state for each data point, then it converts to the density
    matrix representation, and lastly it converts to a sparse Pauli basis
    representation. For 2-qubits, there are 16 axes so the returned quantum
    dataset will be of shape (M x 16) where each datapoint (row) will have
    16 values corresponding to the 16 coefficients in this feature space.

    Args:
        ds (np.ndarray): classical dataset
        feature_map (PauliFeatureMap): feature map to encode the classical
            datapoints to a quantum state

    Returns:
        np.ndarray: quantum Pauli decomposition representation of dataset
        List[str]: Pauli basis axes
    """

    quantum_ds = np.empty((ds.shape[0], 16))
    pauli_axes = [
        'II', 'XI', 'YI', 'ZI', 'IX', 'XX', 'YX', 'ZX', 'IY', 'XY', 'YY', 'ZY', 'IZ', 'XZ', 'YZ','ZZ'
    ]

    for datapoint_idx, datapoint in enumerate(ds):
        # First, get the state vector of our encoded datapoint
        qc = feature_map.bind_parameters(datapoint)
        state_vector = Statevector.from_instruction(qc)

        # Convert the state vector to a density matrix
        density_matrix = DensityMatrix(state_vector)

        # Then decompose the matrix by the Pauli matrices
        pauli_decomp = SparsePauliOp.from_operator(density_matrix)

        # Lastly, construct full representation
        full_pauli_decomp = OrderedDict(zip(pauli_axes, [0.0]*len(pauli_axes)))
        full_pauli_decomp.update(dict(pauli_decomp.to_list()))

        quantum_ds[datapoint_idx][:] = np.array(list(full_pauli_decomp.values()))

    return quantum_ds, pauli_axes


def convert_to_pauli_decomposition(ds: np.ndarray, encoding_fn: Callable) -> Tuple[np.ndarray, List[str]]:
    """
    Given a dataset of shape (M x N) where M is the number of samples and
    N is the number of features, this function returns a 2D array of the
    quantum representation of the dataset given a feature map. This
    representation first computes the state vector of the data-encoded
    quantum state for each data point, then it converts to the density
    matrix representation, and lastly it converts to a sparse Pauli basis
    representation. For 2-qubits, there are 16 axes so the returned quantum
    dataset will be of shape (M x 16) where each datapoint (row) will have
    16 values corresponding to the 16 coefficients in this feature space.

    Args:
        ds (np.ndarray): classical dataset
        feature_map (PauliFeatureMap): feature map to encode the classical
            datapoints to a quantum state

    Returns:
        np.ndarray: quantum Pauli decomposition representation of dataset
        List[str]: Pauli basis axes
    """
    
    x1, x2 = np.hsplit(ds, 2)
    x3 = encoding_fn(x1, x2)

    return np.hstack([fn(x1, x2, x3) for fn in coefficients.values()]), list(coefficients.keys())


def compute_accuracy(labels_ordered: np.ndarray, threshold_idx: int) -> float:
    """
    Given the ordered list of labels, compute the accuracy
    where the desired pattern is all labels of 1 fall on the left
    side of the threshold index and all labels of 0 fall on the right
    side of the threshold index. This accuracy calculation comes from
    equation 7 in [1].

    Args:
        labels_ordered (np.ndarray): list of ordered labels
        threshold_idx (int): threshold of classification boundary

    Returns:
        float: accuracy, in range 0-1.0
    """
    n = labels_ordered.shape[0]
    labels_left = labels_ordered[0: threshold_idx]
    n_plus = np.sum(labels_left == 1)
    n_minus = np.sum(labels_left == 0)

    return (1/n)*(max(n_plus, n_minus) + (n/2)  - min(n_plus,  n_minus))


def compute_pauli_decomposition(ds: np.ndarray, feature_map: PauliFeatureMap, axes: Union[int, List[int]]) -> np.ndarray:
    """
    Given a dataset of shape (M x N) where M is the number of samples and
    N is the number of features, this function returns an array of the
    coefficients for the specified axes of the dataset.

    Args:
        ds (np.ndarray): classical dataset
        feature_map (PauliFeatureMap): feature map to encode the classical
            datapoints to a quantum state
        axes (int | List[int]): axis indices, 0-15

    Returns:
        np.ndarray: coefficients of all datapoints in dataset
    """
    # Make list of axes if just a single value was passed
    axes = [axes] if isinstance(axes, int) else axes

    # Allocate space for coefficients for all datapoints
    coeffs = np.empty((ds.shape[0], len(axes)))

    # Get basis matrices
    basis_mats = [basis_matrices_2qubit[axis] for axis in axes]

    for datapoint_idx, datapoint in enumerate(ds):
        # Get density matrix representation of data
        qc = feature_map.bind_parameters(datapoint)
        density_matrix = DensityMatrix.from_instruction(qc)

        # Project onto axes to get coefficient(s)
        for i in np.arange(len(axes)):
            coeff = np.trace(basis_mats[i].dot(density_matrix)) / 2**2
            coeffs[datapoint_idx][i] = coeff.real
        

    return coeffs
