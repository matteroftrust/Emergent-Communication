from cupy.random import random_integers
import cupy as np
import pickle as pkl
from scipy.stats import zscore
from datetime import datetime as dt

from .utils import generate_item_pool, generate_negotiation_time, print_all, print_status, discount, flatten, unpack, get_weight_grad, printProgressBar


def zscore2(arr):
    zscored = zscore(arr)
    if np.isnan(zscored).any():
        return arr
    return zscored


class Action:
    """
    A negotiation message.
    """

    def __init__(self, terminate, utterance, proposal, id=None):
        self.proposed_by = id
        self.terminate = bool(terminate)  # should be fixed somewhere becaouse it gets [[val]] in StateBatch
        self.utterance = utterance
        self.proposal = proposal.astype(int)

    def __str__(self):
        return 'Action prop_by: {}, term: {}, utter: {}, prop: {}'.format(self.proposed_by, self.terminate, self.utterance, self.proposal)

    __repr__ = __str__

    def is_valid(self, item_pool):
        return not (self.proposal > item_pool).any()


class HiddenState:
    def __init__(self, array):
        self.hs = np.array(array)
        if self.hs.shape != (100,):
            raise ValueError('Hidden state dimensions are wrong. Received: {} with shape {}'.format(type(array), np.array(array).shape))

    def __repr__(self):
        return 'hidden_state'


class StateBatch:
    """
    Stores trajectories and rewards of both agents.

    trajectories contain lists of actions of different lengths
    reward[i] is a reward in i-th trajectory
    item_pools contain a list of item_pools

    would be smart to be able to apply discounted_rewards function to whole matrix without checking lenghths
    """

    def __init__(self, max_trajectory_len=10, item_num=3,
                 hidden_state_size=100, ids=[0, 1]):
        # trajectory_len = np.ceil(max_trajectory_len/2).astype(int)
        self.trajectories_0 = []
        self.trajectories_1 = []
        self.item_pools = []
        self.rewards_0 = []
        self.rewards_1 = []
        self.hidden_states_0 = []
        self.hidden_states_1 = []
        self.ns = []
        self.utilities_0 = []
        self.utilities_1 = []

    def append(self, i, n, trajectory, rewards, item_pool, hidden_states, utilities, max_trajectory_len=10):

        hidden_states = [HiddenState(hs) for hs in hidden_states]
        trajectory_odd = trajectory[::2]
        trajectory_even = trajectory[:1][::2]

        hidden_states_odd = hidden_states[::2]
        hidden_states_even = hidden_states[:1][::2]

        hidden_states_odd.reverse()
        hidden_states_even.reverse()
        trajectory_odd.reverse()
        trajectory_even.reverse()

        is_first_0 = trajectory[0].proposed_by == 0

        if is_first_0:  # agent 0 gets odd
            self.trajectories_0.append(trajectory_odd)
            self.trajectories_1.append(trajectory_even)
            self.hidden_states_0.append(hidden_states_odd)
            self.hidden_states_1.append(hidden_states_even)

        else:  # == 1  agent - gets even
            self.trajectories_0.append(trajectory_even)
            self.trajectories_1.append(trajectory_odd)
            self.hidden_states_0.append(hidden_states_even)
            self.hidden_states_1.append(hidden_states_odd)

        self.rewards_0.append(rewards[not is_first_0])
        self.rewards_1.append(rewards[is_first_0])

        self.item_pools.append(item_pool)
        self.ns.append(n)  # do we even need this now? maybe for regularization later?

        self.utilities_0.append(utilities[0])
        self.utilities_1.append(utilities[1])

    def concatenate(self, batch):
        self.trajectories_0.append(batch.trajectories_0)
        self.trajectories_1.append(batch.trajectories_1)
        self.item_pools.append(batch.item_pools)
        self.rewards_0.append(batch.rewards_0)
        self.rewards_1.append(batch.rewards_1)
        self.ns.append(batch.ns)
        self.utilities_0.append(batch.utilities_0)
        self.utilities_1.append(batch.utilities_1)

    def numpize(self):
        keys = self.__dict__.keys()
        for key in keys:
            self.__dict__[key] = np.array(self.__dict__[key])

    @classmethod
    def compute_discounted_rewards(self, trajectory, reward, discount_factor=0.99):

        input = np.ones(len(trajectory), reward) * reward
        return discount(input, discount_factor), trajectory

    def save_log(self):
        pass

    def convert_for_training(self, baseline, prosocial):
        # TODO: this whole code needs a person equipped with a brain
        rewards_0 = []
        rewards_1 = []

        self.rewards = [self.rewards_0, self.rewards_1]
        # print('rewardss before\n', self.rewards)

        # subtract baseline
        baseline = .7 * baseline + .3 * np.mean(self.rewards, 1)

        if prosocial:
            self.rewards[0] = self.rewards[0] - baseline[0]
            if not all(reward == 0 for reward in self.rewards[0]):
                self.rewards[0] = zscore2(self.rewards[0])

        else:
            self.rewards[0] = self.rewards[0] - baseline[0]
            self.rewards[1] = self.rewards[1] - baseline[0]

            # standardize rewards
            if not all(reward == 0 for reward in self.rewards[0]):
                self.rewards[0] = zscore2(self.rewards[0])
            if not all(reward == 0 for reward in self.rewards[1]):
                self.rewards[1] = zscore2(self.rewards[1])

        for i in range(len(self.ns)):

            if prosocial:
                reward_0 = self.rewards[0][i]
                reward_1 = self.rewards[0][i]
            else:
                reward_0 = self.rewards[0][i]
                reward_1 = self.rewards[1][i]

            t_0_len = len(self.trajectories_0[i])
            t_1_len = len(self.trajectories_1[i])

            input_0 = np.ones(t_0_len) * reward_0
            input_1 = np.ones(t_1_len) * reward_1

            trajectory_rewards_0 = discount(input_0)
            trajectory_rewards_1 = discount(input_1)

            rewards_0.append(trajectory_rewards_0)
            rewards_1.append(trajectory_rewards_1)

        rewards_0 = flatten(rewards_0)
        rewards_1 = flatten(rewards_1)

        trajectories_0 = flatten(self.trajectories_0)
        trajectories_1 = flatten(self.trajectories_1)

        y_proposal_0 = np.array([elem.proposal for elem in trajectories_0])
        y_proposal_1 = np.array([elem.proposal for elem in trajectories_1])

        y_termination_0 = np.array([elem.terminate for elem in trajectories_0])
        y_termination_1 = np.array([elem.terminate for elem in trajectories_1])

        y_utterance_0 = np.array([elem.utterance for elem in trajectories_0])
        y_utterance_1 = np.array([elem.utterance for elem in trajectories_1])

        x_0 = flatten(self.hidden_states_0)
        x_1 = flatten(self.hidden_states_1)

        x_0 = unpack(x_0)
        x_1 = unpack(x_1)

        print_all('This goes to reinfoce:')
        print_all('what is the shape of x0 {} yterm0 {} yprop0 {} r0 {}'.format(x_0.shape, y_termination_0.shape, y_proposal_0.shape, rewards_0.shape))
        print_all('what is the shape of x1 {} yterm1 {} yprop0 {} r1 {}'.format(x_1.shape, y_termination_1.shape, y_proposal_1.shape, rewards_1.shape))

        rewards = [rewards_0, rewards_1]  # TODO should be change if prosocial
        # print('rewardss after\n', rewards)
        return x_0, x_1, y_termination_0, y_termination_1, y_proposal_0, y_proposal_1, y_utterance_0, y_utterance_1, rewards


class Game:

    def __init__(self, agents, batch_size, test_batch_size, episode_num, filename, item_num=3, prosocial=False, test_every=50):
        self.rounds = []
        self.i = 0
        self.agents = agents  # list of agents
        self.stats = None
        self.scores = np.zeros(len(self.agents))

        self.item_num = item_num
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.episode_num = episode_num
        self.prosocial = prosocial
        self.test_every = test_every

        self.filename = filename

    def get_agent(self, id):
        for agent in self.agents:
            if agent.id == id:
                return agent
        return None

    def play(self, save_as=''):
        results = []

        if self.prosocial:  # if prosocial we do need only a prosocial reward R = R_A + R_B
            baseline = np.zeros(1)
        else:
            baseline = np.zeros(2)

        for i in range(self.episode_num):
            # weights.append(self.agents[0].termination_policy.model.get_weights())
            if i % self.test_every == 0:  # TODO remember it should be 50!
                test_batch = self.tests()  # experiment statistics
                results.append([i, test_batch])
                printProgressBar(i, self.episode_num, prefix='Progress:', suffix='Complete {} / {}'.format(i, self.episode_num), length=50)

            # print_status('\n### Starting episode {} out of {} ###\n'.format(i, self.episode_num))
            batch = self.next_episode()
            # print_all('match_item_pool: {} \n batch_negotiations: {} \n batch_rewards'.format(batch.item_pool, batch_negotiations, batch_rewards))

            baseline = self.reinforce(batch, baseline)
        with open('results/{}.pkl'.format(self.filename), 'wb') as handle:
            pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def next_episode(self, test=False):
        batch = StateBatch()
        # print('proposal policy weights', self.agents[0].proposal_policy.models[0].get_weights()[:5])
        # TODO whould be faster to generate data here
        for i in range(self.batch_size):
            # print_status('Starting batch {}'.format(i))
            # beginning of new round. item pool and utility funcions generation
            item_pool = generate_item_pool()
            negotiation_time = generate_negotiation_time()
            for agent in self.agents:
                agent.generate_util_fun()

            item_pool, negotiations, rewards, n, hidden_states = self.negotiations(item_pool, negotiation_time, test=test)

            batch.append(i, n, trajectory=negotiations, rewards=rewards, item_pool=item_pool,
                         hidden_states=hidden_states, utilities=[self.get_agent(0).utilities, self.get_agent(1).utilities])

        return batch

    def negotiations(self, item_pool, n, test=False):
        action = Action(False, self.agents[0].utterance_policy.dummy, self.agents[0].proposal_policy.dummy)  # dummy action TODO how should it be instantiated
        # should it be chosen randomly?
        # rand_0_or_1 = random_integers(0, 1)
        rand_0_or_1 = 0
        proposer = self.agents[rand_0_or_1]
        hearer = self.agents[1 - rand_0_or_1]
        negotiations = []
        hidden_states = []
        # print_status('\nnew negotiation round:\nitem_pool: {}\nagent {} utility {}\nagent {} utility {}\n'.format(item_pool, proposer.id, proposer.utilities, hearer.id, hearer.utilities))

        for t in range(n):
            proposer, hearer = hearer, proposer  # each negotiation round agents switch roles

            termination_true = t == 0

            context = np.concatenate((item_pool, proposer.utilities))
            action, hidden_state = proposer.propose(context, action.utterance, action.proposal, termination_true=termination_true,
                                                    test=test, item_pool=item_pool)  # if communication channel is closed utterance is a dummy
            negotiations.append(action)
            hidden_states.append(hidden_state)
            # print('Round {}:\nproposer {} proposal {} termination {} utterance {}'.format(t, action.proposed_by, action.proposal, action.terminate, action.utterance))

            if action.terminate or not action.is_valid(item_pool):  # that is a bit weird but should work.
                # print_status('i guest thats where it stops', t, n, 'term', action.terminate, 'valid', action.is_valid(item_pool))
                # n = t + 1
                break  # if terminate then negotiations are over

        rewards = self.compute_rewards(item_pool, negotiations[-2:], proposer, hearer)
        # print_status('negotiations finished.\nagent {} reward {}\nagent {} reward {}'.format(proposer.id, rewards[0], hearer.id, rewards[1]))
        return item_pool, negotiations, rewards, n, hidden_states

    def compute_rewards(self, item_pool, actions, proposer, hearer):
        """
        Method for generating rewards.
        actions contains 2 actions, previous one (with proposal agents agreed on) and last one to check if action is valid.
        """
        if len(actions) == 1:  # TODO this is probably wrong
            reward_proposer = 0
            reward_hearer = 0
        elif actions[1].is_valid(item_pool) and actions[1].terminate:  # if proposal is valid and terminated
            reward_proposer = np.dot(proposer.utilities, item_pool - actions[0].proposal)
            reward_hearer = np.dot(hearer.utilities, actions[0].proposal)
        else:
            reward_proposer = 0
            reward_hearer = 0
        # print_status('what are the rewards?', reward_proposer, reward_hearer)
        return [reward_proposer, reward_hearer]

    def tests(self):
        """
        Runs 5 test batches without training.
        """
        test_batch = StateBatch()
        for i in range(5):
            batch = self.next_episode(test=True)
            test_batch.concatenate(batch)
        return test_batch

    def reinforce(self, batch, baseline):
        x_0, x_1, y_termination_0, y_termination_1, y_proposal_0, y_proposal_1, y_utterance_0, y_utterance_1, rewards = batch.convert_for_training(baseline, self.prosocial)
        if sum(rewards[0]) == 0 or sum(rewards[1]) == 0:  # TODO this is wrong but it breaks if rewards are 0 and gradient vanishes
            return baseline
        if len(x_0) == 0:
            print('No data for reinforce')
            return baseline

        agent_0 = self.agents[0]
        agent_1 = self.agents[1]

        # standardize rewards
        if self.prosocial:
            # TODO
            pass
        else:
            rewards[0] = zscore2(rewards[0])
            rewards[1] = zscore2(rewards[1])
        # print('all that she wants:\n----------------------------------------------\n')
        # msgs = ['x_0', 'x_1', 'y_termination_0', 'y_termination_1', 'y_proposal_0', 'y_proposal_1', 'rewards_0', 'rewards_1']
        # ars = [x_0[:10], x_1[:10], y_termination_0, y_termination_1, y_proposal_0, y_proposal_1, rewards[0], rewards[1]]
        # for msg, ar in zip(msgs, ars):
        #     print(msg, ar)
        # print('----------------------------------------')
        # print('gradient', get_weight_grad(agent_0.termination_policy.model, x_0, y_termination_0))
        agent_0.termination_policy.train(x_0, y_termination_0, rewards[0])
        agent_1.termination_policy.train(x_1, y_termination_1, rewards[1])

        agent_0.proposal_policy.train(x_0, y_proposal_0, rewards[0])
        agent_1.proposal_policy.train(x_1, y_proposal_1, rewards[1])

        agent_0.utterance_policy.train(x_0, y_utterance_0, rewards[0])
        agent_1.utterance_policy.train(x_1, y_utterance_1, rewards[1])

        # print('Reinforce done!!!!!')

        # TODO:
        # for core model training it would be smater to move encoders to the model so we dont have to store 1500 values each round
        return baseline
