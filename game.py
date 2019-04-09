from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

from numpy.random import random_integers, poisson
import itertools
import numpy as np
import random as rand


class Action:
    """
    A negotiation message.
    """

    def __init__(self, terminate, utterance, proposal, id=None):
        self.proposed_by = id
        if terminate:
            self.terminate = True
            self.utterance = None
            self.proposal = None
        else:
            self.terminate = False
            self.utterance = utterance
            self.proposal = proposal

    def is_valid(self, item_pool):
        if (self.proposal > item_pool).any():
            return False
        return True


class Game:

    def __init__(self, end, agents, settings):
        self.end = end
        self.rounds = []
        self.i = 0
        self.agents = agents  # list of agents
        self.stats = None
        self.scores = np.zeros(len(self.agents))

        # we might need something like this here:
        self.settings = {
            'linguistic_channel': settings['linguistic_channel'] if 'linguistic_channel' in settings else True,

        }

    def play(self):
        while True:
            print('Starting round {} out of {}'.format(self.i, self.end))
            self.i += 1
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
                return int(out)

    def negotiations(self, item_pool):
        action = Action(False, None, None)  # dummy action TODO how should it be instantiated

        # should it be chosen randomly?
        rand_0_or_1 = random_integers(0, 1)
        agent_1 = self.agents[rand_0_or_1]
        agent_2 = self.agents[1 - rand_0_or_1]

        n = self.generate_negotiation_time()

        for t in range(n):
            if t % 2:  # agents alter their roles
                proposer, hearer = agent_1, agent_2
            else:
                proposer, hearer = agent_2, agent_1
            context = np.concatenate(item_pool, proposer.utilities)

            action = proposer.propose(context, action.utterance, action.proposal)

            if action.terminate:
                # assign rewards
                reward_proposer, reward_hearer = self.compute_rewards(item_pool, action, proposer, hearer)


                proposer.reward(reward_proposer)
                hearer.reward(reward_hearer)
                return

            # assign rewards based on last proposal if agents don't get to any agreement
            if agent_1.id == action.proposed_by:
                proposer, hearer = agent_1, agent_2
            else:
                proposer, hearer = agent_2, agent_1

            reward_proposer = np.dot(proposer.utilities, action.proposal)
            reward_hearer = np.dot(hearer.utilities, item_pool - action.proposal)

            proposer.reward(reward_proposer)
            hearer.reward(reward_hearer)

            return

    def next_round(self):

        if self.i == self.end:
            print('End of the game.')
            return False

        # beginning of new round. item pool and utility funcions generation
        item_pool = self.generate_item_pool()
        self.generate_util_functions()

        self.negotiations(item_pool)

        return True

    def compute_rewards(self, item_pool, action, proposer, hearer):
        """
        Method for generating rewards. Might be more clear to convert it to a class.
        """
        if action.is_valid(item_pool):
            reward_proposer = np.dot(proposer.utilities, action.proposal)
            reward_hearer = np.dot(hearer.utilities, item_pool - action.proposal)
        else:
            reward_proposer = 0
            reward_hearer = 0
        return reward_proposer, reward_hearer
