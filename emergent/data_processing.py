import pickle as pkl
import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="darkgrid")


def load_results(filename):
    with open('results/' + filename, 'rb') as handle:
        return pkl.load(handle)


def run_analysis(filename):
    data = load_results(filename)

    rewards_0 = []
    rewards_1 = []

    for state_batch in data:
        state_batch = state_batch[1]

        compute_batch(state_batch)

        rewards_0.append(state_batch.mean_st_reward_0)
        rewards_1.append(state_batch.mean_st_reward_1)

    X = np.linspace(1, len(rewards_0), len(rewards_0))
    plt.figure(figsize=(15, 10))
    plt.legend(['Agent 0', 'Agent 1'])
    plt.xlabel('training iterations')
    plt.ylabel('relative reward')
    plt.title('Relative rewards')
    sns.lineplot(x=X, y=rewards_0)
    sns.lineplot(x=X, y=rewards_1)
    plt.savefig('figs/rewards.png')


def numpize(batch):
    keys = batch.__dict__.keys()
    for key in keys:
        batch.__dict__[key] = np.array(batch.__dict__[key])


def compute_batch(batch):
    try:
        batch.numpize()
    except:
        numpize(batch)

    batch.rewards_0 = batch.rewards[0]
    batch.rewards_1 = batch.rewards[1]

    batch.item_pools = batch.item_pools.reshape(-1, 3)
    batch.rewards_0 = batch.rewards_0.reshape(-1)
    batch.rewards_1 = batch.rewards_1.reshape(-1)
    batch.utilities_0 = batch.utilities_0.reshape(-1, 3)
    batch.utilities_1 = batch.utilities_1.reshape(-1, 3)
    batch.max_rewards_0 = np.sum(batch.item_pools * batch.utilities_0, axis=1)
    batch.max_rewards_1 = np.sum(batch.item_pools * batch.utilities_1, axis=1)

    batch.st_rewards_0 = np.nan_to_num(batch.rewards_0 / batch.max_rewards_0, 0)  # if item_pools * utilities = [0, 0, 0] then this breaks so it shouldt be 0 but 1
    batch.st_rewards_1 = np.nan_to_num(batch.rewards_1 / batch.max_rewards_1, 0)

    is_nan_0 = np.isnan(batch.st_rewards_0)
    is_nan_1 = np.isnan(batch.st_rewards_1)
    batch.st_rewards_0[is_nan_0] = 0
    batch.st_rewards_1[is_nan_1] = 0

    batch.mean_st_reward_0 = np.mean(batch.st_rewards_0)
    batch.mean_st_reward_1 = np.mean(batch.st_rewards_1)

    batch.trajectory_len = []
    for t1, t2 in zip(batch.trajectories_0, batch.trajectories_1):
        for tt1, tt2 in zip(t1, t2):
            batch.trajectory_len.append(len(tt1) + len(tt2))

    batch.avg_trajectory_len = np.mean(batch.trajectory_len)
