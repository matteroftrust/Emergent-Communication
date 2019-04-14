from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
# from keras.preprocessing import sequence

from numpy.random import random_integers
import itertools
import numpy as np

from game import Action


class EmbeddingTable:
    def __init__(self, input_size, hidden_state_size):
        self.model = Sequential([Embedding(input_dim=input_size, output_dim=hidden_state_size)])

    def embed(self, input):
        return self.model.predict(input)


class Policy:
    def __init__(self, settings):
        pass


class TerminationPolicy(Policy):
    """
    This is a binary decision, and we parametrise πterm as a single feedforward layer,
    with the hidden state as input, followed by a sigmoid function, to represent the probability of termination.
    """
    def __init__(self, hidden_state_size, entropy_reg=0.05):
        # single feedforward layer with sigmoid function
        self.model = Sequential([
            Dense(1, input_shape=(hidden_state_size,)),
            Activation('sigmoid')
        ])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',  # TODO these are random, needs to be checked
                           metrics=['accuracy'])

    def forward(self, hidden_state):
        confidence = self.model.predict(hidden_state)
        return confidence >= 0.5

    def train(self):
        pass


class UtterancePolicy(Policy):
    """
    This is parametrised by an LSTM, which takes the hidden state of the agent as the initial hidden state.
    For the first timestep, a dummy symbol is fed in as input;
    subsequently, the model prediction from the previous timestep is fed in as input at the next timestep,
    in order to predict the next symbol
    """
    def __init__(self, hidden_state_size, entropy_reg=0.001):
        self.model = Sequential([
            LSTM(100, input_shape=(hidden_state_size, 1))
        ])
        self.model.compile(optimizer='adam',
                           loss='mse',  # TODO these are random, needs to be checked
                           metrics=['accuracy'])

    def forward(self, hidden_state):
        utterance = self.model.predict(hidden_state)
        return utterance

    def train(self):
        pass


class ProposalPolicy(Policy):
    """
    This is parametrised by 3 separate feedforward neural networks, one for each item type,
    which each take as input ht and output a distribution over {0...5} indicating the proposal for that item
    """
    def __init__(self, hidden_state_size, item_num=3, entropy_reg=0.05):
        self.item_num = item_num
        self.models = []
        for _ in range(self.item_num):
            model = Sequential([
                LSTM(100, input_shape=(hidden_state_size, 1))
            ])
            model.compile(optimizer='adam',
                          loss='mse',  # TODO these are random, needs to be checked
                          metrics=['accuracy'])
            self.models.append(model)

    def forward(self, hidden_state):
        proposal = []
        for i in range(self.item_num):
            single_proposal = self.models[i].predict(hidden_state)
            proposal.append(single_proposal)
        return proposal

    def train(self):
        pass


class Agent:

    id_generator = itertools.count()

    def __init__(self, lambda_term, lambda_prop, lambda_utt, hidden_state_size, vocab_size, dim_size, utter_len):
        self.id = next(self.id_generator)
        self.lambda_term = lambda_term
        self.lambda_prop = lambda_prop
        self.lambda_utt = lambda_utt

        # policies
        self.term_policy = TerminationPolicy(hidden_state_size)
        self.utter_policy = UtterancePolicy(hidden_state_size)
        self.prop_policy = ProposalPolicy(hidden_state_size)

        self.utter_len = utter_len
        self.vocab_size = vocab_size

    def __str__(self):
        return 'agent {}'.format(self.id)

    @classmethod
    def create_agents(self, n, *args, **kwargs):
        agents = [Agent(*args, **kwargs) for _ in range(n)]
        return agents

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
        return Action(False, np.zeros(10), np.zeros(3), self.id)

    def reward(self, reward):
        pass
