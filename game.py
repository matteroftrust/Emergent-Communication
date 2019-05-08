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

    def __str__(self):
        return 'Action prop_by: {}, term: {}, utter: {}, prop: {}'.format(self.proposed_by, self.terminate, self.utterance, self.proposal)

    def is_valid(self, item_pool):
        return not (self.proposal > item_pool).any()


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

            item_pool, negotiations, rewards = self.negotiations(item_pool, negotiation_time)
            batch_item_pool.append(item_pool)
            batch_negotiations.append(negotiations)
            batch_rewards.append(rewards)
            # remember about random order while adding stuff to batch_negotiations nad batch_rewards

        return batch_item_pool, batch_negotiations, batch_rewards

    def negotiations(self, item_pool, n):
        action = Action(False, np.zeros(self.agents[0].utterance_len), np.zeros(self.item_num))  # dummy action TODO how should it be instantiated
        # should it be chosen randomly?
        rand_0_or_1 = random_integers(0, 1)
        proposer = self.agents[rand_0_or_1]
        hearer = self.agents[1 - rand_0_or_1]
        negotiations = []

        for t in range(n):
            proposer, hearer = hearer, proposer  # each negotiation round agents switch roles

            context = np.concatenate((item_pool, proposer.utilities))
            action = proposer.propose(context, action.utterance, action.proposal)  # if communication channel is closed utterance is a dummy
            negotiations.append(action)
            # print('we are in t: {} and action is {}'.format(t, action))

            # print('action.terminate {}, action.isvalid {}'.format(action.terminate, action.is_valid(item_pool)))

            if action.terminate or not action.is_valid(item_pool):  # that is a bit weird but should work.
                break  # if terminate then negotiations are over

        # assign rewards
        reward_proposer, reward_hearer = self.compute_rewards(item_pool, action, proposer, hearer)
        return item_pool, negotiations, [reward_proposer, reward_hearer]

    def compute_rewards(self, item_pool, action, proposer, hearer):
        """
        Method for generating rewards. Might be more clear to convert it to a class.
        """
        if action.is_valid(item_pool) or not action.terminate:  # if proposal is valid and terminated
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
