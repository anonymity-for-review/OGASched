"""
Some auxiliary functions.
"""
import functools
import random
import time
import numpy as np
from ogasched.utils.logging import logging


def sample_from_bernoulli(trials, prob_threshold):
    """
    Sample from Bernoulli distribution.
    """
    samples = np.repeat(0, trials)
    for i in range(trials):
        samples[i] = 1 if random.random() <= prob_threshold else 0
    return samples


def sample_from_uniform(low, high, size):
    """
    Sample from Uniform distribution.
    """
    return np.random.uniform(low, high, size)


def sample_from_uniform_int(low, high, size):
    """
    Sample integers from Uniform distribution.
    """
    return np.random.randint(low, high, size)


def get_run_time(algo):
    """
    A wrapper that records the run time of algo.
    :param algo:
    :return:
    """
    @functools.wraps(algo)
    def wrapper(*args, **kwargs):
        s = time.time()
        res = algo(*args, **kwargs)
        e = time.time()
        logging.info("%s overall run time: %fs\n", algo.__name__, (e - s))
        return res
    return wrapper
