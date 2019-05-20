from numpy.random import random_integers, poisson
from .settings import load_settings

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


def print_status(message):
    if getattr(project_settings, 'prompt', 'status') == 'status':
        print(message)


def print_all(message):
    if getattr(project_settings, 'prompt', 'status') in ['status', 'all']:
        print(message)


def validation(func):
    """
    Wrapper for validation.
    """
    def function_wrapper(*args, **kwargs):
        print('this is validation!!!!!! decorator')
        print(project_settings.validation)
        if project_settings.validation:
            is_valid, msg = func(*args, **kwargs)
            if not is_valid:
                print('Validation failed\n' + msg)
    return function_wrapper
