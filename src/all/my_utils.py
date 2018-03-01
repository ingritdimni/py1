import numpy as np
import pandas as pd


def test_split():
    df_x = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'B': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'h']})
    df_y = df_x['B']

    x_80, x_20, (p80, p20) = split_input(df_x, 0.8, random=True, return_indices=True, seed=1)
    assert(np.array_equal(df_y[p20], pd.Series(['i', 'f'], name='B')))

    reduced_x, cross_val_x, reduced_y, cross_val_y = split_inputs(df_x, df_y, split_ratio=0.8, seed=1)
    assert(np.array_equal(df_y[p20], cross_val_y))


def split_input(x, split_ratio, random=True, seed=None, return_indices=False):
    """ return reduced_x, cross_val_x and optionally involved_indices
    reduced_x has size split_ratio * x.shape[0]
    cross_val_x has size (1-split_ratio) * x.shape[0]
    involved_indices (optional) is a list of 2 elements: first and second set of involved_indices"""
    if seed: np.random.seed(seed)
    assert(0. <= split_ratio <= 1.)

    if random:
        # shuffle inputs
        p = np.random.permutation(x.shape[0])
        if isinstance(x, np.ndarray):
            x = x[p]
        elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.iloc[p]
        else:
            raise TypeError("input type should be ndarray or DataFrame or Series")
    else:
        p = np.array(range(x.shape[0]))

    # split our training all into 2 sets
    n = int(x.shape[0] * split_ratio)
    cross_val_x = x[n:]
    reduced_x = x[:n]

    if return_indices:
        return reduced_x, cross_val_x, [p[:n], p[n:]]
    return reduced_x, cross_val_x


def split_inputs(x, y, split_ratio, random=True, seed=None, return_indices=False):
    """ same as split_input, with two inputs x and y to split"""
    if seed: np.random.seed(seed)
    assert(x.shape[0] == y.shape[0])

    if random:
        # shuffle inputs
        p = np.random.permutation(x.shape[0])
        if isinstance(x, np.ndarray):
            x = x[p]
            y = y[p]
        elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.iloc[p]
            y = y.iloc[p]
        else:
            raise TypeError("input type should be ndarray or DataFrame or Series")
    else:
        p = np.array(range(x.shape[0]))

    # split our training all into 2 sets
    n = int(x.shape[0] * split_ratio)
    cross_val_x = x[n:]
    reduced_x = x[:n]
    cross_val_y = y[n:]
    reduced_y = y[:n]

    if return_indices:
        return reduced_x, cross_val_x, reduced_y, cross_val_y, [p[:n], p[n:]]
    return reduced_x, cross_val_x, reduced_y, cross_val_y


if __name__ == "__main__":
    test_split()
