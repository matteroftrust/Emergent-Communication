import numpy as np
import pickle as pkl

from .utils import generate_item_pool, generate_negotiation_time, discounts, flatten, zscore2, printProgressBar

try:
    import cupy
    from .utils import zscore2_gpu
except ImportError:
    print('cupy not found.')

class Action:
    """
    A negotiation message.
    """

    def __init__(self, terminate, utterance, proposal, id=None):
        self.proposed_by = id
        self.terminate = bool(terminate)  # TODO should be fixed somewhere becaouse it gets [[val]] in StateBatch
        self.utterance = utterance
        self.proposal = proposal.astype(int) if not np.isnan(proposal[0]) else proposal  # TODO super ugly

    def __str__(self):
        return 'Action prop_by: {}, term: {}, utter: {}, prop: {}'.format(self.proposed_by, self.terminate, self.utterance, self.proposal)

    __repr__ = __str__

    def is_valid(self, item_pool):
        if self.terminate:
            return True
        return not (self.proposal > item_pool).any()


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
        self.trajectories_0 = []
        self.trajectories_1 = []
        self.item_pools = []
        self.rewards = [[], []]
        self.hidden_states_0 = []
        self.hidden_states_1 = []
        self.ns = []
        self.utilities_0 = []
        self.utilities_1 = []

    def copy(self):

        def np_copy(arr):
            return list(np.copy(arr))

        batch = StateBatch()
        batch.trajectories_0 = np_copy(self.trajectories_0)
        batch.trajectories_1 = np_copy(self.trajectories_1)
        batch.item_pool = np_copy(self.item_pools)
        batch.rewards = np_copy(self.rewards)
        batch.hidden_states_0 = np_copy(self.hidden_states_0)
        batch.hidden_states_0 = np_copy(self.hidden_states_1)
        batch.ns = np_copy(self.ns)
        batch.utilities_0 = np_copy(self.utilities_0)
        batch.utilities_1 = np_copy(self.utilities_1)

        return batch

    def append(self, i, n, trajectory, rewards, item_pool, hidden_states, utilities, max_trajectory_len=10):

        trajectory_even = trajectory[::2]  # even and odd indexwise so arr[0] is even and arr[1] odd
        trajectory_odd = trajectory[1:][::2]

        # print('what comes to append', len(hidden_states))
        hidden_states_even = hidden_states[::2]
        hidden_states_odd = hidden_states[1:][::2]
        # print('whats after', hidden_states_even.shape, hidden_states_odd.shape)

        hidden_states_odd.reverse()
        hidden_states_even.reverse()
        trajectory_odd.reverse()
        trajectory_even.reverse()

        is_first_0 = trajectory[0].proposed_by == 0

        if is_first_0:  # agent 0 gets odd
            self.trajectories_0.append(trajectory_even)
            self.trajectories_1.append(trajectory_odd)
            self.hidden_states_0.append(hidden_states_even)
            self.hidden_states_1.append(hidden_states_odd)

        else:  # == 1  agent - gets even
            self.trajectories_0.append(trajectory_odd)
            self.trajectories_1.append(trajectory_even)
            self.hidden_states_0.append(hidden_states_odd)
            self.hidden_states_1.append(hidden_states_even)

        self.rewards[0].append(rewards[0])  # no need to check because its a dict
        self.rewards[1].append(rewards[1])

        self.item_pools.append(item_pool)
        self.ns.append(n)  # do we even need this now? maybe for regularization later?

        self.utilities_0.append(utilities[0])
        self.utilities_1.append(utilities[1])

    def concatenate(self, batch):
        self.trajectories_0.append(batch.trajectories_0)
        self.trajectories_1.append(batch.trajectories_1)
        self.item_pools.append(batch.item_pools)
        self.rewards[0].append(batch.rewards[0])
        self.rewards[1].append(batch.rewards[1])

        self.ns.append(batch.ns)
        self.utilities_0.append(batch.utilities_0)
        self.utilities_1.append(batch.utilities_1)

    def numpize(self):
        keys = self.__dict__.keys()
        for key in keys:
            self.__dict__[key] = np.array(self.__dict__[key])

    def convert_for_training(self, baseline, prosocial):
        # TODO: this whole code needs a person equipped with a brain

        rewards = self.rewards.copy()
        rewards_0 = []
        rewards_1 = []

        # subtract baseline
        baseline = .7 * baseline + .3 * np.mean(self.rewards, 1)

        if prosocial:
            rewards[0] = rewards[0] - baseline[0]
            if not all(reward == 0 for reward in rewards[0]):
                rewards[0] = zscore2(rewards[0])

        else:
            rewards[0] = rewards[0] - baseline[0]
            rewards[1] = rewards[1] - baseline[0]

            # standardize rewards
            if not all(reward == 0 for reward in rewards[0]):
                rewards[0] = zscore2(rewards[0])
            if not all(reward == 0 for reward in rewards[1]):
                rewards[1] = zscore2(rewards[1])

        for i in range(len(self.ns)):

            if prosocial:
                reward_0 = rewards[0][i]
                reward_1 = rewards[0][i]
            else:
                reward_0 = rewards[0][i]
                reward_1 = rewards[1][i]

            trajectory_rewards_0 = discounts(reward_0, len(self.trajectories_0[i]))
            trajectory_rewards_1 = discounts(reward_1, len(self.trajectories_1[i]))

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

        # print('what is the shape of x0 {} yterm0 {} yprop0 {} r0 {}'.format(x_0.shape, y_termination_0.shape, y_proposal_0.shape, rewards_0.shape))
        # print('what is the shape of x1 {} yterm1 {} yprop0 {} r1 {}'.format(x_1.shape, y_termination_1.shape, y_proposal_1.shape, rewards_1.shape))

        rewards = [rewards_0, rewards_1]  # TODO should be change if prosocial
        # print('rewardss after\n', rewards)
        return x_0, x_1, y_termination_0, y_termination_1, y_proposal_0, y_proposal_1, y_utterance_0, y_utterance_1, rewards

    def convert_for_training_gpu(self, baseline, prosocial):
        # TODO: this whole code needs a person equipped with a brain

        self.numpize()
        rewards = self.rewards.copy()
        rewards_0 = []
        rewards_1 = []

        # subtract baseline
        baseline = np.array(baseline)
        baseline = .7 * baseline + .3 * cupy.mean(self.rewards, 1)

        if prosocial:
            rewards[0] = rewards[0] - baseline[0]
            if not all(reward == 0 for reward in rewards[0]):
                rewards[0] = zscore2(rewards[0])

        else:
            rewards[0] = rewards[0] - baseline[0]
            rewards[1] = rewards[1] - baseline[0]

            # standardize rewards
            if not all(reward == 0 for reward in rewards[0]):
                rewards[0] = zscore2(rewards[0])
            if not all(reward == 0 for reward in rewards[1]):
                rewards[1] = zscore2(rewards[1])

        for i in range(len(self.ns)):

            if prosocial:
                reward_0 = rewards[0][i]
                reward_1 = rewards[0][i]
            else:
                reward_0 = rewards[0][i]
                reward_1 = rewards[1][i]

            trajectory_rewards_0 = discounts(reward_0, len(self.trajectories_0[i]))
            trajectory_rewards_1 = discounts(reward_1, len(self.trajectories_1[i]))

            rewards_0.append(trajectory_rewards_0)
            rewards_1.append(trajectory_rewards_1)

        rewards_0 = flatten(rewards_0)
        rewards_1 = flatten(rewards_1)

        trajectories_0 = flatten(self.trajectories_0)
        trajectories_1 = flatten(self.trajectories_1)

        y_proposal_0 = cupy.array([elem.proposal for elem in trajectories_0])
        y_proposal_1 = cupy.array([elem.proposal for elem in trajectories_1])

        y_termination_0 = cupy.array([elem.terminate for elem in trajectories_0])
        y_termination_1 = cupy.array([elem.terminate for elem in trajectories_1])

        y_utterance_0 = cupy.array([elem.utterance for elem in trajectories_0])
        y_utterance_1 = cupy.array([elem.utterance for elem in trajectories_1])

        x_0 = flatten(self.hidden_states_0)
        x_1 = flatten(self.hidden_states_1)

        # print('what is the shape of x0 {} yterm0 {} yprop0 {} r0 {}'.format(x_0.shape, y_termination_0.shape, y_proposal_0.shape, rewards_0.shape))
        # print('what is the shape of x1 {} yterm1 {} yprop0 {} r1 {}'.format(x_1.shape, y_termination_1.shape, y_proposal_1.shape, rewards_1.shape))

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

        if self.prosocial:  # if prosocial we need only a prosocial reward R = R_A + R_B
            baseline = np.zeros(1)
        else:
            baseline = np.zeros(2)

        for i in range(self.episode_num):
            if i % self.test_every == 0:
                test_batch = self.tests()  # experiment statistics
                results.append([i, test_batch])
                printProgressBar(i, self.episode_num, prefix='Progress:', suffix='Complete {} / {}'.format(i, self.episode_num), length=50)

            batch = self.next_episode()
            baseline = self.reinforce(batch, baseline)

        with open('results/{}.pkl'.format(self.filename), 'wb') as handle:
            pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def next_episode(self, test=False):
        batch = StateBatch()
        # print('proposal policy weights', self.agents[0].proposal_policy.models[0].get_weights()[:5])
        # TODO whould be faster to generate data here
        for i in range(self.batch_size):
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
        contexts = {  # context doesnt change during negotiation so can be outside of the loop
            proposer.id: proposer.context_encoder(np.concatenate((item_pool, proposer.utilities))),
            hearer.id: hearer.context_encoder(np.concatenate((item_pool, hearer.utilities)))
        }
        negotiations = []
        hidden_states = []
        # print_status('\nnew negotiation round:\nitem_pool: {}\nagent {} utility {}\nagent {} utility {}\n'.format(item_pool, proposer.id, proposer.utilities, hearer.id, hearer.utilities))
        termination_true = True  # during the first round an agent can't terminate

        for t in range(n):
            action, hidden_state = proposer.propose(contexts[proposer.id], action.utterance, action.proposal, termination_true=termination_true,
                                                    test=test, item_pool=item_pool)  # if communication channel is closed utterance is a dummy
            negotiations.append(action)
            hidden_states.append(hidden_state)
            # print('Round {}:\nproposer {} proposal {} termination {} utterance {}'.format(t, action.proposed_by, action.proposal, action.terminate, action.utterance))

            if not action.is_valid(item_pool):
                break

            if action.terminate:  # so its valid and terminated
                reward_proposer = np.dot(proposer.utilities, item_pool - negotiations[-2].proposal)
                reward_hearer = np.dot(hearer.utilities, negotiations[-2].proposal)
                rewards = {proposer.id: reward_proposer, hearer.id: reward_hearer}

                return item_pool, negotiations, rewards, n, hidden_states

            proposer, hearer = hearer, proposer  # each negotiation round agents switch roles
            termination_true = False

        return item_pool, negotiations, {0: 0, 1: 0}, n, hidden_states

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
        import timeit, functools

        batch_cpu, baseline_cpu = batch.copy(), np.copy(baseline)
        batch_gpu, baseline_gpu = batch.copy(), np.copy(baseline)

        t = timeit.Timer(functools.partial(batch_cpu.convert_for_training, baseline_cpu, self.prosocial))
        print('time CPU!!!', t.timeit(10))
        t = timeit.Timer(functools.partial(batch_gpu.convert_for_training_gpu, baseline_gpu, self.prosocial))
        print('time GPU!!!', t.timeit(10))

        x_0, x_1, y_termination_0, y_termination_1, y_proposal_0, y_proposal_1, y_utterance_0, y_utterance_1, rewards = batch.convert_for_training(baseline, self.prosocial)



        if sum(rewards[0]) == 0 or sum(rewards[1]) == 0:  # TODO this is wrong but it breaks if rewards are 0 and gradient vanishes
            return baseline
        if len(x_0) == 0:
            print('No data for reinforce')
            return baseline

        agent_0 = self.agents[0]
        agent_1 = self.agents[1]

        # print('what goes to train000 ', x_0.shape, y_proposal_0.shape, rewards[0].shape)

        agent_0.termination_policy.train(x_0, y_termination_0, rewards[0])
        agent_1.termination_policy.train(x_1, y_termination_1, rewards[1])

        rm_ids = []
        # print('iterate opver this malak', y_proposal_0.shape)

        # TODO: hey is there a way to do that better?
        for i, y in enumerate(y_proposal_0):
            if np.isnan(y[0]):
                rm_ids.append(i)

        y_proposal_0 = np.delete(y_proposal_0, rm_ids, axis=0)
        y_utterance_0 = np.delete(y_utterance_0, rm_ids, axis=0)
        rewards[0] = np.delete(rewards[0], rm_ids)
        x_0 = np.delete(x_0, rm_ids, axis=0)

        rm_ids = []
        for i, y in enumerate(y_proposal_1):
            if np.isnan(y[0]):
                rm_ids.append(i)

        y_proposal_1 = np.delete(y_proposal_1, rm_ids, axis=0)
        y_utterance_1 = np.delete(y_utterance_1, rm_ids, axis=0)
        rewards[1] = np.delete(rewards[1], rm_ids)
        x_1 = np.delete(x_1, rm_ids, axis=0)
        # print('what goes to train000 ', x_0.shape, y_proposal_0.shape, rewards[0].shape)
        # print('what goes to train111 ', x_1.shape, y_proposal_1.shape, rewards[1].shape)

        agent_0.proposal_policy.train(x_0, y_proposal_0, rewards[0])
        agent_1.proposal_policy.train(x_1, y_proposal_1, rewards[1])

        agent_0.utterance_policy.train(x_0, y_utterance_0, rewards[0])
        agent_1.utterance_policy.train(x_1, y_utterance_1, rewards[1])


        # TODO:
        # for core model training it would be smater to move encoders to the model so we dont have to store 1500 values each round
        return baseline
