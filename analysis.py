import pickle as pkl
import itertools
import matplotlib.pyplot as plt
import numpy as np

from emergent.game import HiddenState


def load_results(filename):
    with open('results/' + filename, 'rb') as handle:
        return pkl.load(handle)


def run_analysis(filename):
    data = load_results(filename)

    rewards_0 = []
    rewards_1 = []

    for state_batch in data:
        state_batch = state_batch[1]
        sb_rewards_0 = list(itertools.chain(*state_batch.rewards_0))
        sb_rewards_1 = list(itertools.chain(*state_batch.rewards_1))
        mean_rewards_0 = np.mean(sb_rewards_0)
        mean_rewards_1 = np.mean(sb_rewards_1)

        rewards_0.append(mean_rewards_0)
        rewards_1.append(mean_rewards_1)

    X = np.linspace(1, len(rewards_0), len(rewards_0))
    plt.plot(X, rewards_0, 'r')
    plt.plot(X, rewards_1, 'b')
    plt.savefig('figs/rewards.png')
