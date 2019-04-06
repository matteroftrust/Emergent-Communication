import itertools
import numpy as np
import random as rand
from numpy.random import random_integers, poisson


class Action:
    """
    A negotiation message.
    """

    def __init__(self, terminate, message, proposal, id=None):
        self.proposed_by = id
        if terminate:
            self.terminate = True
            self.message = None
            self.proposal = None
        else:
            self.terminate = False
            self.message = message
            self.proposal = proposal

    def is_valid(self, item_pool):
        if (self.proposal > item_pool).any():
            return False
        return True


class Agent:

    id_generator = itertools.count()

    def __init__(self, lambda_term, lambda_prop, lambda_utt):
        self.id = next(self.id_generator)
        self.lambda_term = lambda_term
        self.lambda_prop = lambda_prop
        self.lambda_utt = lambda_utt

        self.initiate_model()

    def __str__(self):
        return 'agent {}'.format(self.id)

    @classmethod
    def create_agents(self, n, *args, **kwargs):
        agents = [Agent(*args, **kwargs) for _ in range(n)]
        return agents

    def initiate_model(self):
        """
        Neural network initialization.
        """
        # self.hidden_state =
        pass

    def generate_util_fun(self):
        """
        Generate utility function which specifies rewards for each item.
        """
        while True:
            out = random_integers(0, 10, 3)
            if list(out) != [0, 0, 0]:
                self.utterance = out
                return out

    def propose(self, context, utterance, proposal):

        # hidden_state
        return Action(False, None, None, self.id)

    def reward(self, reward):
        pass


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

            action = proposer.propose(None, None, None)

            if action.terminate:
                # assign rewards
                if action.is_valid(item_pool):
                    reward_proposer = np.dot(proposer.utterance, action.proposal)
                    reward_hearer = np.dot(hearer.utterance, item_pool - action.proposal)
                else:
                    reward_proposer = 0
                    reward_hearer = 0

                proposer.reward(reward_proposer)
                hearer.reward(reward_hearer)
                return

            # assign rewards based on last proposal if agents don't get to any agreement
            if agent_1.id == action.proposed_by:
                proposer, hearer = agent_1, agent_2
            else:
                proposer, hearer = agent_2, agent_1

            reward_proposer = np.dot(proposer.utterance, action.proposal)
            reward_hearer = np.dot(hearer.utterance, item_pool - action.proposal)

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
