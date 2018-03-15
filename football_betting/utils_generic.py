import numpy as np
import pandas as pd


def contain_nan(el):
    """ recursive implementation to look for nan values """
    try:
        for e in el:
            if contain_nan(e):
                return True
    except TypeError:
        return np.isnan(el)
    return False


def get_array(array):
    """ return the array of pandas or numpy object """
    if isinstance(array, pd.DataFrame) or isinstance(array, pd.Series):
        return array.values
    if isinstance(array, np.ndarray):
        return array
    raise TypeError('array should be pd.DataFrame or pd.Series or np.ndarray')


def printv(message_importance, cur_verbose_level, *args):
    """ print only is verbose level is high enough """
    if cur_verbose_level >= message_importance:
        print(*args)


if __name__ == "__main__":
    printv(1, 0, "coucou", "ça va ?", 2)
