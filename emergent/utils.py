from .settings import load_settings
from numpy.random import random_integers, poisson

import numpy as np

project_settings, _, _ = load_settings()


def generate_item_pool():
    return random_integers(0, 5, 3)


def generate_negotiation_time():
    """
    Generate negotiation time ampled from truncated Poisson distribution.
    TODO it should be truncated Poisson but this one is not I guess! Needs to be checked!
    """
    while True:
        out = poisson(7, 1)
        if out >= 4 and out <= 10:
            return int(out)


def print_all(message):
    if project_settings.prompt == 'all':
        print(message)


def print_status(message):
    if project_settings.prompt in ['status', 'all']:
        print(message)


def validation(func):
    """
    Wrapper for validation.
    """
    def function_wrapper(*args, **kwargs):
        if project_settings.validation:
            is_valid, msg = func(*args, **kwargs)
            if not is_valid:
                print('###### Validation failed ######\n' + msg)
    return function_wrapper


def discount(r, gamma=0.99, standardize=False):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801]
    taken from https://github.com/breeko/Simple-Reinforcement-Learning-with-Tensorflow
    """
    discounted = np.array([val * (gamma ** i) for i, val in enumerate(r)])
    if standardize:
        discounted -= np.mean(discounted)
        discounted /= np.std(discounted)
    return discounted
