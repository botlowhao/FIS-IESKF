import numpy as np

def custom_mf2(x, a, b):
    """
    Custom fuzzy membership generator(custom_mf2).

    Parameters
    ----------
    x : 1d array
        Independent variable.
    a : float
        'Ceiling', where the function begins falling from 1.
    b : float
        'Foot', where the function reattains zero.

    Returns
    -------
    y : 1d array

    """
    assert a <= b, 'a <= b is required.'

    y = np.ones(len(x))

    # Values less than a
    idx = x < a
    y[idx] = 0

    # Values between a and (a + b) / 2
    idx = np.logical_and(a <= x, x < (a + b) / 2.)
    y[idx] = 1 - 2. * ((x[idx] - a) / (b - a)) ** 2.

    # Values between (a + b) / 2 and b
    idx = np.logical_and((a + b) / 2. <= x, x <= b)
    y[idx] = 2. * ((x[idx] - b) / (b - a)) ** 2.

    # Values greater than or equal to b
    idx = x >= b
    y[idx] = 0

    return y

def custom_mf3(x, a, b):
    """
    Custom fuzzy membership generator(custom_mf3).

    Parameters
    ----------
    x : 1d array
        Independent variable.
    a : float
        'Foot', where the function begins to climb from zero.
    b : float
        'Ceiling', where the function levels off at 1.

    Returns
    -------
    y : 1d array

    """
    assert a <= b, 'a <= b is required.'
    y = np.ones(len(x))

    # Values less than or equal to a
    idx = x <= a
    y[idx] = 0

    # Values between a and (a + b) / 2
    idx = np.logical_and(a <= x, x <= (a + b) / 2.)
    y[idx] = 2. * ((x[idx] - a) / (b - a)) ** 2.

    # Values between (a + b) / 2 and b
    idx = np.logical_and((a + b) / 2. <= x, x <= b)
    y[idx] = 1 - 2. * ((x[idx] - b) / (b - a)) ** 2.

    # Values greater than b
    idx = x > b
    y[idx] = 0

    return y


def Inverted_gbellmf(x, a, b, c):
    return 1 - (1. / (1. + np.abs((x - c) / a) ** (2 * b)))