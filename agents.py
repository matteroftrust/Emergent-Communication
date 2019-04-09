from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

from numpy.random import random_integers, poisson
import itertools
import numpy as np
import random as rand


class Policy:
    def __init__(self, settings):
        pass


class TerminationPolicy(Policy):
    """
    This is a binary decision, and we parametrise πterm as a single feedforward layer,
    with the hidden state as input, followed by a sigmoid function, to represent the probability of termination.
    """"
    def __init__():
        # single feedforward layer
        # sigmoid function
        pass


class UtterancePolicy(Policy):
    """
    This is parametrised by an LSTM, which takes the hidden state of the agent as the initial hidden state.
    For the first timestep, a dummy symbol is fed in as input;
    subsequently, the model prediction from the previous timestep is fed in as input at the next timestep,
    in order to predict the next symbol
    """
    pass


class ProposalPolicy(Policy):
    """
    This is parametrised by 3 separate feedforward neural networks, one for each item type,
    which each take as input ht and output a distribution over {0...5} indicating the proposal for that item
    """
    pass


class Agent:

    id_generator = itertools.count()

    def __init__(self, lambda_term, lambda_prop, lambda_utt, settings):
        self.id = next(self.id_generator)
        self.lambda_term = lambda_term
        self.lambda_prop = lambda_prop
        self.lambda_utt = lambda_utt

        # policies
        self.term_policy = TerminationPolicy()
        self.utter_policy = UtterancePolicy()
        self.prop_policy = ProposalPolicy()

        self.initiate_model(settings)

    def __str__(self):
        return 'agent {}'.format(self.id)

    @classmethod
    def create_agents(self, n, *args, **kwargs):
        agents = [Agent(*args, **kwargs) for _ in range(n)]
        return agents

    def initiate_model(self, settings):
        """
        Neural network initialization.
        """
        self.model = Sequential()
        self.model.add(Embedding(settings['vocab_size'], settings['dim_size']))
        self.model.add(LSTM(100))  # TODO is it also dim_size?
        self.model.compile(optimizer='adam', loss='mse')

    def generate_util_fun(self):
        """
        Generate utility function which specifies rewards for each item.
        """
        while True:
            out = random_integers(0, 10, 3)
            if list(out) != [0, 0, 0]:
                self.utilities = out
                return out

    def propose(self, context, utterance, proposal):

        # hidden_state
        return Action(False, None, None, self.id)

    def reward(self, reward):
        pass
