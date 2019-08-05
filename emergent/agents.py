from numpy.random import random_integers
import itertools
import numpy as np

from .game import Action
from .models import AllInOneModel


class Agent:

    id_generator = itertools.count()

    def __init__(self, hidden_state_size, vocab_size, dim_size, utterance_len, discount_factor, learning_rate,
                 proposal_channel, linguistic_channel, lambda_termination, lambda_utterance, lambda_proposal):
        self.id = next(self.id_generator)
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.utterance_len = utterance_len

        self.allinone = AllInOneModel(hidden_state_size, 6, 6, 3)

    def __str__(self):
        return 'agent {}'.format(self.id)

    @classmethod
    def create_agents(self, n, *args, **kwargs):
        agents = [Agent(*args, **kwargs) for _ in range(n)]
        return agents

    @property
    def dummy_utterance(self):
        return np.ones(self.utterance_len)

    @property
    def dummy_proposal(self):
        return np.ones(3)

    def generate_util_fun(self):
        """
        Generate utility function which specifies rewards for each item.
        """
        while True:
            out = random_integers(0, 10, 3)
            if list(out) != [0, 0, 0]:
                self.utilities = out
                return out

    def propose(self, context, utterance, proposal, test, termination_true=False):
        termination, utterance, proposal, y = self.allinone.predict(context, utterance, proposal, test=test)

        if termination_true:
            termination = False
        hidden_state = None

        action = Action(terminate=termination, utterance=np.array(utterance), proposal=np.array(proposal), id=self.id)

        return action, hidden_state, y
