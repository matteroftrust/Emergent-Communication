from numpy.random import random_integers
import numpy as np

from .settings import load_settings
from .utils import generate_item_pool, generate_negotiation_time, print_all, print_status, discount, flatten

project_settings, agent_settings, game_settings = load_settings()


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

    def __init__(self, batch_size=game_settings.batch_size,
                 max_trajectory_len=10, item_num=game_settings.item_num,
                 hidden_state_size=agent_settings.hidden_state_size, ids=[0, 1]):
        # trajectory_len = np.ceil(max_trajectory_len/2).astype(int)
        self.trajectories_0 = []
        self.trajectories_1 = []
        self.item_pools = []
        self.rewards_0 = []
        self.rewards_1 = []
        self.hidden_states_0 = []
        self.hidden_states_1 = []
        self.ns = []

    def append(self, i, n, trajectory, rewards, item_pool, hidden_states, max_trajectory_len=10):
        # trajectory = trajectory)  # so the last action is first, that will make the discounted rewards computations easier

        # TODO: this is suuuper ugly
        # to be changed earlier in negotiations which should return separate arrays for agents

        # print('what comes from episode?\n')
        # print('trajectory type {} shape {}'.format(type(trajectory), np.array(trajectory).shape))
        # print('hidden_states shape {}'.format(np.array(hidden_states).shape))
        #
        # print('in statebatch what is hidden state', np.array(hidden_states).shape)
        # print('in statebatch wat is hidden state self', len(self.hidden_states_0))

        hidden_states = [HiddenState(hs) for hs in hidden_states]
        print('what are hidden_states here///////??', hidden_states)
        trajectory_odd = trajectory[::2]
        trajectory_even = trajectory[:1][::2]

        hidden_states_odd = hidden_states[::2]
        hidden_states_even = hidden_states[1:][::2]

        hidden_states_odd.reverse()
        hidden_states_even.reverse()
        trajectory_odd.reverse()
        trajectory_even.reverse()

        print('what is in hs odd shape {} type {}'.format(np.array(hidden_states_odd).shape, type(hidden_states_odd)))
        print('what is in hs even shape {} is empty list? {}'.format(np.array(hidden_states_even).shape, hidden_states_even == []))

        # print('statbatc in append hssod: {} hsseven:: {}'.format(np.array(hidden_states_odd).shape, np.array(hidden_states_even).shape))

        is_first_0 = trajectory[0].proposed_by == 0

        if is_first_0:  # agent 0 gets odd
            self.trajectories_0.append(trajectory_odd)
            self.trajectories_1.append(trajectory_even)
            self.hidden_states_0.append(hidden_states_odd)
            self.hidden_states_1.append(hidden_states_even)

            # self.trajectories_0.append(np.flip(trajectory_odd))
            # self.trajectories_1.append(np.flip(trajectory_even))
            # self.hidden_states_0.append(np.flip(hidden_states_odd))
            # self.hidden_states_1.append(np.flip(hidden_states_even))

        else:  # == 1
            self.trajectories_0.append(trajectory_even)
            self.trajectories_1.append(trajectory_odd)
            self.hidden_states_0.append(hidden_states_even)
            self.hidden_states_1.append(hidden_states_odd)

            # self.trajectories_0.append(np.flip(trajectory_even))
            # self.trajectories_1.append(np.flip(trajectory_odd))
            # self.hidden_states_0.append(np.flip(hidden_states_even))
            # self.hidden_states_1.append(np.flip(hidden_states_odd))

        self.rewards_0.append(rewards[not is_first_0])
        self.rewards_1.append(rewards[is_first_0])

        self.item_pools.append(item_pool)
        self.ns.append(n)  # do we even need this now?
        # self.ids = np.array()
        # print('in statebatch wat is hidden state self after', len(self.hidden_states_0))

    @classmethod
    def compute_discounted_rewards(self, trajectory, reward, discount_factor=0.99):
        # TODO: the one below is ugly but isnan doesnt work

        # mask = [action is not None for action in trajectory]
        # trajectory = trajectory[mask]
        # trajectory = trajectory[~np.isnan(trajectory, dtype=Action)]  # removing None vals

        input = np.ones(len(trajectory), reward) * reward
        return discount(input, discount_factor), trajectory

    def save_log(self):
        pass

    def convert_for_training(self):
        # x_0 = [np.array([])]
        # x_1 = np.array([])
        y_proposal_0 = []
        y_proposal_1 = []

        rewards_0 = []
        rewards_1 = []

        # for i in range(len(self.ns)):
        #     len()
        #     trajectory_0, discount_rewards_0 = self.compute_discounted_rewards(self.trajectories_0[i], self.rewards_0[i])
        #     trajectory_1, discount_rewards_1 = self.compute_discounted_rewards(self.trajectories_1[i], self.rewards_1[i])
        #     np.append(x_0, trajectory_0)
        #     np.append(x_1, trajectory_1)
        for i in range(len(self.ns)):
            t_0_len = len(self.trajectories_0[i])
            t_1_len = len(self.trajectories_1[i])
            reward_0 = self.rewards_0[i]
            reward_1 = self.rewards_1[i]

            input_0 = np.ones(t_0_len) * reward_0
            input_1 = np.ones(t_1_len) * reward_1

            trajectory_rewards_0 = discount(input_0)
            trajectory_rewards_1 = discount(input_1)

            rewards_0.append(trajectory_rewards_0)
            rewards_1.append(trajectory_rewards_1)

        rewards_0 = np.array(rewards_0).flatten()
        rewards_1 = np.array(rewards_1).flatten()
        # y_0 = np.flatten(self.hidden_states_0)
        # y_1 = np.flatten(self.hidden_states_1)
        # print('straight from statebatch x0 {} x1{}'.format(np.array(self.hidden_states_0).shape, np.array(self.hidden_states_1).shape))

        # print('what aare the shapes? x0 {} x1 {}'.format(x0_shape, x1_shape))
        print('is it about to go through?')
        # print('trajectories_0', self.trajectories_0)
        # print('y trajectories', self.trajectories_0)
        # print('1 trajectories', self.trajectories_1)

        def print_trajectory(t, name):
            print('\n{}\n'.format(name))
            for elem in t:
                print(elem, type(elem))

        # print_trajectory(np.array(self.trajectories_0).flatten(), 'traje 0')
        # print_trajectory(self.trajectories_1, 'traje 1')

        y_termination_0 = flatten(self.trajectories_0)
        y_termination_1 = flatten(self.trajectories_1)
        # y_termination_0 = np.array([t[0].terminate for t in self.trajectories_0])
        # y_termination_1 = np.array([t[0].terminate for t in self.trajectories_1])

        # print_trajectory(y_termination_0, 'y term 0')

        print('went through!')

        # print('this i x_0 but whyyyy', type(x_0), type(x_0[0]))

        # y_termination_0 = np.array([trajectory.terminate for trajectory in np.array(self.trajectories_0).flatten()])
        # y_termination_1 = np.array([trajectory.terminate for trajectory in np.array(self.trajectories_1).flatten()])

        # TODO: rewards need reguralization
        # print('checking dimensions in convert_for_training')
        # print('agent0, x_0 {} y_0 {}, rewards_0 {}'.format(x_0.shape, y_termination_0.shape, rewards_0.shape))
        # print('agent0, x_1 {} y_1 {}, rewards_1 {}'.format(x_1.shape, y_termination_1.shape, rewards_1.shape))
        x_0 = flatten(self.hidden_states_0)
        x_1 = flatten(self.hidden_states_1)

        # print_trajectory(x_0, 'wtf xo')

        def unpack(arr):
            new_arr = []
            for hs in arr:
                new_arr.append(hs.hs)
            return np.array(new_arr)

        x_0 = unpack(x_0)
        x_1 = unpack(x_1)

        print('what is the shave of x0 {} y0 {}'.format(x_0.shape, y_termination_0.shape))
        print('what is the shave of x1 {} y1 {}'.format(x_1.shape, y_termination_1.shape))


        return x_0, y_termination_0, rewards_0

    """def convert_for_training_old(self):
        # TODO this should return stuff divided into to sets for two users
        # divide data into 2 agents
        agent_ids = list(set([trajectory.proposed_by for trajectory in self.trajectories[0] if trajectory is not None]))

        # for i in range(len(self.ns)):

        x = self.hidden_states[0][:self.ns[0]]
        # print('what are you trajectory', self.trajectories[0])
        y = np.array([action.terminate for action in self.trajectories[0][:self.ns[0]]])
        y = np.reshape(y, (self.ns[0], 1))
        # print('shapes of convert!!!! x {}'.format(x.shape))
        # print('shapes of convert!!!! y {}'.format(y.shape))
        sample_weight = np.array([self.rewards[0][0] for _ in range(self.ns[0])])
        sample_weight = sample_weight.astype(int)
        return x, y, sample_weight"""


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
        # print('batchsize in statebatch which is from game init: {}'.format(batch_size))

    def play(self):
        for i in range(self.episode_num):
            # weights.append(self.agents[0].termination_policy.model.get_weights())
            if i % 50:
                self.tests()  # experiment statistics

            print_status('\n### Starting episode {} out of {} ###\n'.format(i, self.episode_num))
            batch = self.next_episode()
            # print_all('match_item_pool: {} \n batch_negotiations: {} \n batch_rewards'.format(batch.item_pool, batch_negotiations, batch_rewards))

            self.reinforce(batch)

    def next_episode(self):
        batch = StateBatch()
        for i in range(self.batch_size):

            # beginning of new round. item pool and utility funcions generation
            print_status('Starting batch {}'.format(i))
            item_pool = generate_item_pool()
            negotiation_time = generate_negotiation_time()
            for agent in self.agents:
                agent.generate_util_fun()

            item_pool, negotiations, rewards, n, hidden_states = self.negotiations(item_pool, negotiation_time)

            if n == 1:
                print('this is when an agent terminates after dummy message, so negotiations[0].terminate should be True. Is it true? {}'.format(negotiations[0].terminate))
                # TODO thats actually a problem we should solve. If agent terminates after dummy message we dont have a hidden state for the second agent
                continue
            batch.append(i, n, negotiations, rewards, item_pool, hidden_states)

        # TODO remember about random order while adding stuff to batch_negotiations nad batch_rewards

        return batch

    def negotiations(self, item_pool, n):
        action = Action(False, np.zeros(self.agents[0].utterance_len), np.zeros(self.item_num))  # dummy action TODO how should it be instantiated
        # should it be chosen randomly?
        rand_0_or_1 = random_integers(0, 1)
        proposer = self.agents[rand_0_or_1]
        hearer = self.agents[1 - rand_0_or_1]
        negotiations = []
        hidden_states = []

        for t in range(n):
            proposer, hearer = hearer, proposer  # each negotiation round agents switch roles

            context = np.concatenate((item_pool, proposer.utilities))
            action, hidden_state = proposer.propose(context, action.utterance, action.proposal)  # if communication channel is closed utterance is a dummy
            negotiations.append(action)
            hidden_states.append(hidden_state)
            # print('what the hell is a hidden state here?????', np.array(hidden_state).shape)
            print_all('we are in t: {} and action is {}'.format(t, action))

            if action.terminate or not action.is_valid(item_pool):  # that is a bit weird but should work.
                n = t + 1
                break  # if terminate then negotiations are over

        # assign rewards

        # print('AFTER NEGOTIATIONS:\nhidden state len {} shape of hs: {} n: {}'.format(len(hidden_states), np.array(hidden_states).shape, n))
        # print('negotiations: neg len: {} neg shape: {}'.format(len(negotiations), np.array(negotiations).shape))
        # print('how does it look like then?', negotiations)

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
        print('what are the rewards? ', reward_proposer, reward_hearer)
        return reward_proposer, reward_hearer

    def tests(self):
        pass

    def reinforce(self, batch):
        x, y, sample_weight = batch.convert_for_training()
        agent = self.agents[0]
        # sample_weight = np.expand_dims(sample_weight, axis=1)
        # TODO what does it mean: sample_weight_mode="temporal" in compile(). If you just mean to use sample-wise weights, make sure your sample_weight array is 1D.
        print_all('Reinforce input shape: x: {} y: {} sample_weight: {}'.format(x.shape, y.shape, sample_weight.shape))
        out = agent.termination_policy.train(x, y, sample_weight)
        print('Reinforce done!!!!!')
        print_all(out)

        # TerminationPolicy takes boolean as y
        # ProposalPolicy takes action.proposal as y
        # UtterancePolicy takes action.utterance as y if utterance channel is on

        # print('weigths', np.sum(agent.termination_policy.model.get_weights()))

        # TODO:
        # for core model training it would be smater to move encoders to the model so we dont have to store 1500 values each round
