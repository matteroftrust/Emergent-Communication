from numpy.random import random_integers, poisson


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
