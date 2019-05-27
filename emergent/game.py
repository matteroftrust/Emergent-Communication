from numpy.random import random_integers
import numpy as np

from .settings import load_settings
from .utils import generate_item_pool, generate_negotiation_time, print_all, print_status

project_settings, agent_settings, game_settings = load_settings()


class Action:
    """
    A negotiation message.
    """

    def __init__(self, terminate, utterance, proposal, id=None):
        self.proposed_by = id
        self.terminate = terminate
        self.utterance = utterance
        self.proposal = proposal.astype(int)

    def __str__(self):
        return 'Action prop_by: {}, term: {}, utter: {}, prop: {}'.format(self.proposed_by, self.terminate, self.utterance, self.proposal)

    __repr__ = __str__

    def is_valid(self, item_pool):
        return not (self.proposal > item_pool).any()


class StateBatch:
    """
    Stores trajectories and rewards of both agents.

    trajectories contain lists of actions of different lengths
    reward[i] is a reward in i-th trajectory
    item_pools contain a list of item_pools

    would be smart to be able to apply discounted_rewards function to whole matrix without checking lenghths
    """

    def __init__(self, batch_size=game_settings.batch_size,
                 max_trajectory_len=10, item_num=game_settings.item_num,
                 hidden_state_size=agent_settings.hidden_state_size):

        self.trajectories = np.empty((batch_size, max_trajectory_len), dtype=Action)
        self.item_pools = np.zeros((batch_size, item_num), dtype='int32')
        self.rewards = np.zeros((batch_size, 2), dtype='int32')
        self.hidden_states = np.zeros((batch_size, max_trajectory_len, hidden_state_size), dtype='float32')

    def append(self, i, n, trajectory, rewards, item_pool, hidden_states, max_trajectory_len=10):
        trajectory = np.flip(trajectory)  # so the last action is first, that will make the discounted rewards computations easier
        self.trajectories[i][:n] = trajectory
        self.rewards[i] = rewards
        self.item_pools[i] = item_pool
        self.hidden_states[i] = hidden_states

    def compute_discounted_rewards(self, discount_factor):
        pass

    def save_log(self):
        pass


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
        print('batchsize in statebatch which is from game init: {}'.format(batch_size))

    def play(self):
        for i in range(self.episode_num):

            if i % 50:
                self.tests()  # experiment statistics

            print_status('\n### Starting episode {} out of {} ###\n'.format(i, self.episode_num))
            batch = self.next_episode()
            # print_all('match_item_pool: {} \n batch_negotiations: {} \n batch_rewards'.format(batch.item_pool, batch_negotiations, batch_rewards))

            self.reinforce(batch)

    def next_episode(self):
        batch = StateBatch()
        print('batch size in next episode {}'.format(self.batch_size))
        for i in range(self.batch_size):

            # beginning of new round. item pool and utility funcions generation
            print_all('Starting batch {}'.format(i))
            item_pool = generate_item_pool()
            negotiation_time = generate_negotiation_time()
            for agent in self.agents:
                agent.generate_util_fun()

            item_pool, negotiations, rewards, n, hidden_states = self.negotiations(item_pool, negotiation_time)

            batch.append(i, n, negotiations, rewards, item_pool, hidden_states)

        print('bath_item_pool {} batch_negotiations {} batch_rewards {}'.format(batch.item_pools.shape, batch.trajectories.shape, batch.rewards.shape))
        print_all('batch trajectory for {}:'.format(i, batch.trajectories[i]))
        # TODO remember about random order while adding stuff to batch_negotiations nad batch_rewards

        return batch

    def negotiations(self, item_pool, n):
        action = Action(False, np.zeros(self.agents[0].utterance_len), np.zeros(self.item_num))  # dummy action TODO how should it be instantiated
        # should it be chosen randomly?
        rand_0_or_1 = random_integers(0, 1)
        proposer = self.agents[rand_0_or_1]
        hearer = self.agents[1 - rand_0_or_1]
        negotiations = []
        hidden_states = np.zeros((10, agent_settings.hidden_state_size), dtype='float32')  # TODO not sure if zeros is the best idea here

        for t in range(n):
            proposer, hearer = hearer, proposer  # each negotiation round agents switch roles

            context = np.concatenate((item_pool, proposer.utilities))
            action, hidden_state = proposer.propose(context, action.utterance, action.proposal)  # if communication channel is closed utterance is a dummy
            negotiations.append(action)
            hidden_states[t] = hidden_state
            # hidden_states.append(hidden_state)
            print_all('we are in t: {} and action is {}'.format(t, action))

            if action.terminate or not action.is_valid(item_pool):  # that is a bit weird but should work.
                n = t + 1
                break  # if terminate then negotiations are over

        # assign rewards
        reward_proposer, reward_hearer = self.compute_rewards(item_pool, action, proposer, hearer)
        return item_pool, negotiations, [reward_proposer, reward_hearer], n, hidden_states

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

    def reinforce(self, batch):
        pass
