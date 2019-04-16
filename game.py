import numpy as np
from numpy.random import random_integers
from utils import generate_item_pool, generate_negotiation_time


class Action:
    """
    A negotiation message.
    """

    def __init__(self, terminate, utterance, proposal, id=None):
        self.proposed_by = id
        self.terminate = terminate
        self.utterance = utterance
        self.proposal = proposal

    def is_valid(self, item_pool):
        if (self.proposal > item_pool).any():
            return False
        return True


class Game:

    def __init__(self, agents, batch_size, test_batch_size, episode_num, item_num=3):
        self.rounds = []
        self.i = 0
        self.agents = agents  # list of agents
        self.stats = None
        self.scores = np.zeros(len(self.agents))

        self.item_num = item_num
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.episode_num = episode_num
        # we might need something like this here:
        # self.settings = {
        #     'linguistic_channel': settings['linguistic_channel'] if 'linguistic_channel' in settings else True,
        # }

    def play(self):
        for i in range(self.episode_num):

            if i % 50:
                self.tests()  # experiment statistics

            print('### Starting episode {} out of {} ###'.format(i, self.episode_num))
            batch_item_pool, batch_negotiations, batch_rewards = self.next_episode()

            self.reinforce(batch_item_pool, batch_negotiations, batch_rewards)

    def negotiations(self, item_pool, n):
        action = Action(False, np.zeros(self.agents[0].utterance_len), np.zeros(self.item_num))  # dummy action TODO how should it be instantiated
        # should it be chosen randomly?
        rand_0_or_1 = random_integers(0, 1)
        agent_1 = self.agents[rand_0_or_1]
        agent_2 = self.agents[1 - rand_0_or_1]

        for t in range(n):
            print('Round {} out of {}'.format(t, n))
            if t % 2:  # agents alter their roles
                proposer, hearer = agent_1, agent_2
            else:
                proposer, hearer = agent_2, agent_1

            context = np.concatenate((item_pool, proposer.utilities))
            action = proposer.propose(context, action.utterance, action.proposal)

            if action.terminate:
                # assign rewards
                reward_proposer, reward_hearer = self.compute_rewards(item_pool, action, proposer, hearer)

                proposer.reward(reward_proposer)
                hearer.reward(reward_hearer)
                break

            # assign rewards based on last proposal if agents don't get to any agreement
            if agent_1.id == action.proposed_by:
                proposer, hearer = agent_1, agent_2
            else:
                proposer, hearer = agent_2, agent_1
            # TODO what is it in the next line? should be removed?
            reward_proposer = np.dot(proposer.utilities, action.proposal)
            reward_hearer = np.dot(hearer.utilities, item_pool - action.proposal)

            proposer.reward(reward_proposer)
            hearer.reward(reward_hearer)
        # TODO both get reward of zero


    def next_episode(self):
        batch_item_pool = []
        batch_negotiations = []
        batch_rewards = []
        for i in range(self.batch_size):

            # beginning of new round. item pool and utility funcions generation
            item_pool = generate_item_pool()
            negotiation_time = generate_negotiation_time()
            for agent in self.agents:
                agent.generate_util_fun()

            self.negotiations(item_pool, negotiation_time)

            # remember about random order while adding stuff to batch_negotiations nad batch_rewards

        return batch_item_pool, batch_negotiations, batch_rewards

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

    def tests(self):
        pass

    def reinforce(self, batch_item_pool, batch_negotiations, batch_rewards):
        pass
