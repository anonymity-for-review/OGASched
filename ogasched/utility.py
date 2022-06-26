"""
Default utilities.
    - linear: y = a * x
    - log: y = a * ln(x+1)
    - reciprocal: y = 1/a - 1/(x + a)
    - poly: y = a * sqrt(x + 1)
"""
import numpy as np

SCALE_NUM = 3


def linear_u(alpha, y):
    return alpha * y


def derivative_linear_u(alpha, y):
    return alpha


def log_u(alpha, y):
    return alpha * np.log(y + 1)


def derivative_log_u(alpha, y):
    return alpha / (y + 1)


def reciprocal_u(alpha, y, scale=SCALE_NUM):
    return 1 / (scale * alpha) - 1 / (y + scale * alpha)


def derivative_reciprocal_u(alpha, y, scale=SCALE_NUM):
    """
    NOTE that alpha should be larger for this utility. Otherwise, the derivative can be very large,
    which could lead to a large regret.
    :param alpha:
    :param y:
    :param scale:
    :return:
    """
    return 1 / (y + scale * alpha) ** 2


def poly_u(alpha, y):
    return alpha * np.sqrt(y + 1) - alpha


def derivative_poly_u(alpha, y):
    return 0.5 * alpha / np.sqrt(y + 1)
