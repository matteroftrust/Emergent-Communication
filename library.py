import itertools
import numpy as np
import random as rand
from numpy.random import random_integers, poisson


class Agent:

    id_generator = itertools.count()

    def __init__(self, lambda_term, lambda_prop, lambda_utt):
        self.id = next(self.id_generator)
        self.lambda_term = lambda_term
        self.lambda_prop = lambda_prop
        self.lambda_utt = lambda_utt

    def __str__(self):
        return 'agent {}'.format(self.id)

    def generate_util_fun(self):
        """
        Generate utility function which specifies rewards for each item.
        """
        while True:
            out = random_integers(0, 10, 3)
            if list(out) != [0, 0, 0]:
                return out

    def propose(self, context, utterance, proposal):
        pass


class Game:

    def __init__(self, end, agents):
        self.end = end
        self.rounds = []
        self.i = 0
        self.agents = agents  # list of agents
        self.stats = None
        self.scores = np.zeros(len(self.agents))

    def play(self):
        i = 0
        while True:
            i += 1
            out = self.next_round()
            if not out:
                break

    def generate_item_pool(self):
        return random_integers(0, 5, 3)

    def generate_util_functions(self):
        """
        Generate new utility functions for all agents.
        """
        for agent in self.agents:
            agent.generate_util_fun()

    def generate_negotiation_time(self):
        """
        Generate negotiation time ampled from truncated Poisson distribution.
        TODO it should be truncated Poisson but this one is not I guess! Needs to be checked!
        """
        while True:
            out = poisson(7, 1)
            if out >= 4 and out <= 10:
                return out

    def negotiations(self, item_pool):
        # should it be chosen randomly?
        rand_0_or_1 = random_integers(0, 1)
        agent_1 = self.agents[rand_0_or_1]
        agent_2 = self.agents[1 - rand_0_or_1]

        n = self.generate_negotiation_time()

        for t in n:
            if t % 2:  # agents alter their roles
                proposer, hearer = agent_1, agent_2
            else:
                proposer, hearer = agent_2, agent_1


    def next_round(self):

        if self.i == self.end:
            print('End of the game.')
            return False

        # beginning of new round. item pool and utility funcions generation
        item_pool = self.generate_item_pool()
        self.generate_util_functions()

        self.negotiations(item_pool)
