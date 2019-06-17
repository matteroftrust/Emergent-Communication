from .settings import load_settings
from numpy.random import random_integers, poisson
from keras import backend as K

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


def flatten(arr):
    return np.array([elem for sublist in arr for elem in sublist])


def unpack(arr):
    """
    unpack values stored in HiddenState.
    """
    new_arr = []
    for hs in arr:
        new_arr.append(hs.hs)
    return np.array(new_arr)


def print_trajectory(t, name):
    print('\n{}\n'.format(name))
    for elem in t:
        print(elem, type(elem))


def convert_to_sparse(arr, n):
    out = np.zeros((len(arr), n), dtype=int)
    for i, row in enumerate(out):
        row[arr[i]] = 1
    return out


def get_weight_grad(model, inputs, outputs):
    #  https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad
