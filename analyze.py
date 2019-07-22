import argparse
import matplotlib.pyplot as plt
import cupy as np
import seaborn as sns

from emergent.data_processing import load_results, compute_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='filename')

    args = parser.parse_args()

    filename = args.__dict__['filename']

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
    plt.axes().set_ylim((0, 1))
    plt.legend(['Agent 0', 'Agent 1'])
    plt.xlabel('test iterations')
    plt.ylabel('relative reward')
    plt.title('Relative rewards')
    sns.lineplot(x=X, y=rewards_0)
    sns.lineplot(x=X, y=rewards_1)
    plt.savefig('figs/{}_rewards.png'.format(filename))
