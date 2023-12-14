import numpy as np
import scipy.stats as ss
import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.model_selection import train_test_split

from typing import Callable, Dict, List, Optional, Tuple


def _make_xor(n_samples: int, seed: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Makes exclusive OR dataset that has an even split of datapoints in class 0
    and class 1 if n_samples if even; otherwise class 1 will have 1 more datapoint. All X
    values will be in the range -1 to 1.

    Args:
        n_samples (int): number of samples to generate
        seed (int): random seed for data sampling.
        shuffle (bool): shuffle data

    Returns:
        np.ndarray: X, the generated samples
        np.ndarray: y, the labels of the samples (0 or 1)
    """
    # Make labels of half class 0 and half class 1
    class_0_size = int(n_samples/2)
    class_1_size = n_samples - class_0_size
    ys = np.array([0]*class_0_size + [1]*class_1_size)
    
    # Generate x data to have close to even number of datapoints in each corner
    tr_size = int(class_0_size/2)
    bl_size = class_0_size - int(class_0_size/2)
    tl_size = int(class_1_size/2)
    br_size = class_1_size - int(class_1_size/2)
    stds = 1.499

    xs = np.array([
        np.hstack([
            ss.truncnorm.rvs(-stds, stds, loc=0.5, scale=1/3, size=tr_size, random_state=seed),
            ss.truncnorm.rvs(-stds, stds, loc=0.5, scale=1/3, size=bl_size, random_state=seed+1)*-1,
            ss.truncnorm.rvs(-stds, stds, loc=0.5, scale=1/3, size=tl_size, random_state=seed+2)*-1,
            ss.truncnorm.rvs(-stds, stds, loc=0.5, scale=1/3, size=br_size, random_state=seed+3)
        ]),
        np.hstack([
            ss.truncnorm.rvs(-stds, stds, loc=0.5, scale=1/3, size=tr_size, random_state=seed+4),
            ss.truncnorm.rvs(-stds, stds, loc=0.5, scale=1/3, size=bl_size, random_state=seed+5)*-1,
            ss.truncnorm.rvs(-stds, stds, loc=0.5, scale=1/3, size=tl_size, random_state=seed+6),
            ss.truncnorm.rvs(-stds, stds, loc=0.5, scale=1/3, size=br_size, random_state=seed+7)*-1,
        ])
    ]).T

    if shuffle:
        np.random.seed(seed+12)
        shuffle_idxs = np.random.choice(np.arange(n_samples), size=n_samples, replace=False)
        return (xs[shuffle_idxs], ys[shuffle_idxs])
    
    else:
        return (xs, ys)
    
    
def _make_exp(n_samples: int, seed: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Makes exponential dataset that has an even split of datapoints in class 0
    and class 1 if n_samples if even; otherwise class 1 will have 1 more datapoint. All X
    values will be in the range -1 to 1.

    Args:
        n_samples (int): number of samples to generate
        seed (int): random seed for data sampling.
        shuffle (bool): shuffle data

    Returns:
        np.ndarray: X, the generated samples
        np.ndarray: y, the labels of the samples (0 or 1)
    """
    # Set numpy's random seed
    np.random.seed(seed)

    # Make labels of half class 0 and half class 1
    ys = np.array([0]*int(n_samples/2) + [1]*(n_samples - int(n_samples/2)))

    # Generate x data
    x1s = np.random.random(n_samples)
    x2s = np.empty((n_samples,))

    for idx, y in enumerate(ys):
        exp_boundary = 0.1*(10.33**x1s[idx])
        x2s[idx] = np.random.uniform(0.0001, exp_boundary) if y == 0 else np.random.uniform(exp_boundary, 0.999)

    xs = 2*np.array([x1s,  x2s]).T - 1


    # Return shuffled or ordered data
    if shuffle:
        np.random.seed(seed+15)
        shuffle_idxs = np.random.choice(np.arange(n_samples), size=n_samples, replace=False)
        return (xs[shuffle_idxs], ys[shuffle_idxs])
    
    else:
        return (xs, ys)


def _make_circles(n_samples: int, seed: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make circles dataset that has an even split of datapoints in class 0 and class 1 if n_samples
    is even; otherwise class 1 will have 1 more datapoint. All X values will be in the range
    -1 to 1.

    Args:
        n_samples (int): number of samples to generate
        seed (int): random seed for data sampling
        shuffle (bool, optional): shuffle data. Defaults to True.

    Returns:
        np.ndarray: X, the generated samples
        np.ndarray: y, the labels of the samples (0 or 1)
    """
    # Generate data
    X, y = sklearn.datasets.make_circles(
        n_samples=n_samples, factor=0.4, noise=0.1, random_state=seed, shuffle=shuffle
    )

    # Scale the x data because the noise can move it outside the (-1, 1) range
    X_scaled = MinMaxScaler(feature_range=(-0.999, 0.999)).fit_transform(X)

    return X_scaled, y


def _make_moons(n_samples: int, seed: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make moons dataset that has an even split of datapoints in class 0 and class 1 if n_samples
    is even; otherwise class 1 will have 1 more datapoint. All X values will be in the range
    -1 to 1.

    Args:
        n_samples (int): number of samples to generate
        seed (int): random seed for data sampling
        shuffle (bool, optional): shuffle data. Defaults to True.

    Returns:
        np.ndarray: X, the generated samples
        np.ndarray: y, the labels of the samples (0 or 1)
    """
    # Generate data
    X, y = sklearn.datasets.make_moons(
        n_samples=n_samples, noise=0.13, random_state=seed, shuffle=shuffle
    )

    # Scale the x data because the noise can move it outside the (-1, 1) range
    X_scaled = MinMaxScaler(feature_range=(-0.999, 0.999)).fit_transform(X)

    return X_scaled, y


def _sort_data(X: np.ndarray,  y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort data such that the labels alternating. For example, the labels of datapoints
    x1, x2, x3, and x4 will be 0, 1, 0, 1. This is useful in ensuring data is randomly
    distributed initially and that subsets are balanced.

    Args:
        X (np.ndarray): 2-dimensional X data
        y (np.ndarray): 1-dimensional array of labels (0 and 1)

    Returns:
        Tuple[np.ndarray, np.ndarray]: sorted X and y arrays
    """
    # Get the indices of the class 0 datapoints and the class 1 datapoints
    class_0_idxs = np.where(y == 0)[0]
    class_1_idxs = np.where(y == 1)[0]

    # Interleave them together
    reindexed = [*sum(zip(class_0_idxs, class_1_idxs),())]

    # Reassign the reindexes/reordered datapoints
    X_sorted = X[reindexed, :]
    y_sorted = y[reindexed]

    return X_sorted, y_sorted


def make_datasets(
        n_samples: int, training_pct: Optional[float] = None, seed: Optional[int] = None
    ) -> Dict:
    """
    Creates the 4 datasets (circles, moons, exp, and xor) which consist of n_samples of 2-dimensional X data
    and the corresponding binary classification labels as y. If `training_pct` is None, X and y data is
    returned for each dataset; otherwise, the datasets returned are split into 'train' and 'test' partitions
    based on the training/test split percentage. 

    Args:
        n_samples (int): number of samples to generate from each dataset distribution
        training_pct (Optional[float], optional): Training/test split percentage. Defaults to None.
        seed (Optional[int], optional): Random seed for the data sampling. Defaults to Optional[int]=None.

    Returns:
        Dict: dictionary of the datasets
    """
    # Make datasets
    circles = _make_circles(n_samples=n_samples, seed=seed)
    moons = _make_moons(n_samples=n_samples, seed=seed)
    xor = _make_xor(n_samples=n_samples, seed=seed)
    exp = _make_exp(n_samples=n_samples, seed=seed)

    if training_pct is None:
        ds = {
            'circles': {'X': circles[0], 'y': circles[1]},
            'exp': {'X': exp[0], 'y': exp[1]},
            'moons': {'X': moons[0], 'y': moons[1]},
            'xor': {'X': xor[0], 'y': xor[1]}
        }

    else:
        c_X_train, c_X_test, c_y_train, c_y_test = train_test_split(
            circles[0], circles[1], test_size=(1-training_pct), random_state=42, stratify=circles[1]
        )
        e_X_train, e_X_test, e_y_train, e_y_test = train_test_split(
            exp[0], exp[1], test_size=(1-training_pct), random_state=42, stratify=exp[1]
        )
        m_X_train, m_X_test, m_y_train, m_y_test = train_test_split(
            moons[0], moons[1], test_size=(1-training_pct), random_state=42, stratify=moons[1]
        )
        x_X_train, x_X_test, x_y_train, x_y_test = train_test_split(
            xor[0], xor[1], test_size=(1-training_pct), random_state=42, stratify=xor[1]
        )
        ds = {
            'circles': {'train': {'X': c_X_train, 'y': c_y_train}, 'test': {'X': c_X_test, 'y': c_y_test}},
            'exp': {'train': {'X': e_X_train, 'y': e_y_train}, 'test': {'X': e_X_test, 'y': e_y_test}},
            'moons': {'train': {'X': m_X_train, 'y': m_y_train}, 'test': {'X': m_X_test, 'y': m_y_test}},
            'xor': {'train': {'X': x_X_train, 'y': x_y_train}, 'test': {'X': x_X_test, 'y': x_y_test}}
        }

    # Sort the datapoints
    for dataset in ds.keys():
        # Get number of subsets of data to loop through
        subsets = ['train', 'test'] if training_pct is not None else ['--']

        for subset in subsets:
            data_ptr = ds[dataset][subset] if training_pct is not None else ds[dataset]
            data_ptr['X'], data_ptr['y'] = _sort_data(data_ptr['X'], data_ptr['y'])


    return ds


def make_adhoc_dataset(
        n_samples: int,
        training_pct: Optional[float] = None,
        cv_num: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Dict:
    """
    Generates a dataset that samples from a ZZ Feature Map distribution.

    Args:
        n_samples (int): number of samples to generate from each dataset distribution
        training_pct (Optional[float], optional): Training/test split percentage. Defaults to None.
        cv_num: (Optional[int], optional): Number of cross fold validations. Defaults to None.
        seed (Optional[int], optional): Random seed for the data sampling. Defaults to Optional[int]=None.

    Returns:
        Dict: dataset
    """
    # Set random seed first
    algorithm_globals.random_seed = seed

    # Get all data at once and then split for each cross-fold validation
    train_data, train_labels, test_data, test_labels = ad_hoc_data(
        training_size=int(n_samples*training_pct*0.5)*cv_num,
        test_size=int(n_samples*(1-training_pct)*0.5)*cv_num,
        n=2,
        gap=0.3,
        one_hot=False
    )

    # Standardize data
    train_data = np.clip(train_data/np.pi - 1, a_min=-0.999, a_max=0.999)
    test_data = np.clip(test_data/np.pi - 1, a_min=-0.999, a_max=0.999)

    # Sort data so the data labels are alternating.
    train_data, train_labels = _sort_data(train_data, train_labels)
    test_data, test_labels = _sort_data(test_data, test_labels)

    # Split into groups for each cross-fold validation if needed
    if cv_num is not None:
        train_data = np.split(train_data, cv_num)
        train_labels = np.split(train_labels, cv_num)
        test_data = np.split(test_data, cv_num)
        test_labels = np.split(test_labels, cv_num)
    
    
    if training_pct is None:
        ds = {'X': train_data, 'y': train_data}
    else:
        ds = {
            'train': {'X': train_data, 'y': train_labels},
            'test': {'X': test_data, 'y': test_labels}
        }

    return ds

