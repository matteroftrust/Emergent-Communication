from numpy.random import random_integers
import numpy as np

from .settings import load_settings
from .utils import generate_item_pool, generate_negotiation_time, print_all, print_status, discount, flatten, unpack, get_weight_grad

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

    @classmethod
    def compute_discounted_rewards(self, trajectory, reward, discount_factor=0.99):

        input = np.ones(len(trajectory), reward) * reward
        return discount(input, discount_factor), trajectory

    def save_log(self):
        pass

    def convert_for_training(self):
        # TODO: rewards need reguralization
        # TODO: this whole code needs a person equipped with a brain

        rewards_0 = []
        rewards_1 = []

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

        rewards_0 = flatten(rewards_0)
        rewards_1 = flatten(rewards_1)

        trajectories_0 = flatten(self.trajectories_0)
        trajectories_1 = flatten(self.trajectories_1)

        y_proposal_0 = np.array([elem.proposal for elem in trajectories_0])
        y_proposal_1 = np.array([elem.proposal for elem in trajectories_1])

        y_termination_0 = np.array([elem.terminate for elem in trajectories_0])
        y_termination_1 = np.array([elem.terminate for elem in trajectories_1])

        x_0 = flatten(self.hidden_states_0)
        x_1 = flatten(self.hidden_states_1)

        x_0 = unpack(x_0)
        x_1 = unpack(x_1)

        print_all('This goes to reinfoce:')
        print_all('what is the shape of x0 {} yterm0 {} yprop0 {} r0 {}'.format(x_0.shape, y_termination_0.shape, y_proposal_0.shape, rewards_0.shape))
        print_all('what is the shape of x1 {} yterm1 {} yprop0 {} r1 {}'.format(x_1.shape, y_termination_1.shape, y_proposal_1.shape, rewards_1.shape))

        print('\nprinting everything that goes to reinforce\n')
        for elem, name in zip([x_0, x_1, y_termination_0, y_termination_1, y_proposal_0, y_proposal_1, rewards_0, rewards_1], ['x_0', 'x_1', 'y_termination_0', 'y_termination_1', 'y_proposal_0', 'y_proposal_1', 'rewards_0', 'rewards_1']):
            print(name, elem, type(elem), '\n')

        return x_0, x_1, y_termination_0, y_termination_1, y_proposal_0, y_proposal_1, rewards_0, rewards_1


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
        print('agent 0 weights:')
        print('termintation policy weights', self.agents[0].termination_policy.model.get_weights()[:5])
        # print('proposal policy weights', self.agents[0].proposal_policy.models[0].get_weights()[:5])
        for i in range(self.batch_size):
            print_status('Starting batch {}'.format(i))
            # beginning of new round. item pool and utility funcions generation
            item_pool = generate_item_pool()
            negotiation_time = generate_negotiation_time()
            for agent in self.agents:
                agent.generate_util_fun()

            item_pool, negotiations, rewards, n, hidden_states = self.negotiations(item_pool, negotiation_time)

            if n == 1:
                print_all('this is when an agent terminates after dummy message, so negotiations[0].terminate should be True. Is it true? {}'.format(negotiations[0].terminate))
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
        return reward_proposer, reward_hearer

    def tests(self):
        pass

    def reinforce(self, batch):
        x_0, x_1, y_termination_0, y_termination_1, y_proposal_0, y_proposal_1, rewards_0, rewards_1 = batch.convert_for_training()
        if sum(rewards_0) == 0 or sum(rewards_1) == 0:  # TODO this is wrong but it breaks if rewards are 0 and gradient vanishes
            return
        if len(x_0) == 0:
            print('No data for reinforce')
            return
        agent_0 = self.agents[0]
        agent_1 = self.agents[1]
        # sample_weight = np.expand_dims(sample_weight, axis=1)
        # TODO what does it mean: sample_weight_mode="temporal" in compile(). If you just mean to use sample-wise weights, make sure your sample_weight array is 1D.
        # print_all('Reinforce input shape: x: {} y: {} sample_weight: {}'.format(x.shape, y.shape, sample_weight.shape))

        # print('grad???')
        # print(get_weight_grad(agent_0.termination_policy.model, x_0, y_termination_0))
        out_0 = agent_0.termination_policy.train(x_0, y_termination_0, rewards_0)
        out_1 = agent_1.termination_policy.train(x_1, y_termination_1, rewards_1)

        # print('train termination for 0 {}'.format(out_0))
        # print('train termination for 1 {}'.format(out_1))

        # out_0 = agent_0.proposal_policy.train(x_0, y_proposal_0, rewards_0)
        # out_1 = agent_1.proposal_policy.train(x_1, y_proposal_1, rewards_1)

        print('Reinforce done!!!!!')


        # TerminationPolicy takes boolean as y
        # ProposalPolicy takes action.proposal as y
        # UtterancePolicy takes action.utterance as y if utterance channel is on

        # print('weigths', np.sum(agent.termination_policy.model.get_weights()))

        # TODO:
        # for core model training it would be smater to move encoders to the model so we dont have to store 1500 values each round
