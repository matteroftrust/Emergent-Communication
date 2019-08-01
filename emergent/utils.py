from .settings import load_settings
from numpy.random import random_integers, poisson
from tensorflow.python.keras import backend as K

import numpy as np
from scipy.stats import zscore

try:
    import cupy
except ImportError:
    print('cupy not found.')


project_settings, _, _ = load_settings()


class Results:
    def __init__(self):
        self.item_pools = []
        self.rewards_0 = []
        self.rewards_1 = []

    def append(self, item_pools, rewards_0, rewards_1):
        self.item_pools


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


def print_all(*args, **kwargs):
    if project_settings.prompt == 'all':
        print(*args, **kwargs)


def print_status(*args, **kwargs):
    if project_settings.prompt in ['status', 'all']:
        print(*args, **kwargs)


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


def discounts(r, length, gamma=0.99):
    return [r * (gamma ** i) for i in range(length)]


def flatten(arr):
    return np.array([elem for sublist in arr for elem in sublist])


def zscore2(arr):
    zscored = zscore(arr)
    if np.isnan(zscored).any():
        return arr
    return zscored


def zscore2_gpu(arr):
    zscored = zscore(arr)
    if cupy.isnan(zscored).any():
        return arr
    return zscored


def print_trajectory(t, name):
    print('\n{}\n'.format(name))
    for elem in t:
        print(elem, type(elem))


def convert_to_sparse(arr, n):
    # TODO: this should be optimized
    arr = arr.astype(int)
    out = np.zeros((len(arr), n), dtype=int)
    for i, row in enumerate(out):
        row[arr[i]] = 1
    return out


def get_weight_grad(model, inputs, outputs):
    #  https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model.targets + model.sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    # taken from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
