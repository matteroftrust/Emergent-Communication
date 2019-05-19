import argparse
import os

from game import Game
from agents import Agent


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Config:
    def __init__(*kwargs):
        pass


if __name__ == '__main__':

    AGENT_SETTINGS = {
        'lambda_termination': 0.05,  # entropy reguralization weight hyperparameter for termination policy
        'lambda_proposal': 0.05,  # entropy reguralization weight hyperparameter for proposal policy
        'lambda_utterance': 0.001,  # entropy reguralization weight hyperparameter for linguistic utterance policy
        'hidden_state_size': 100,
        'vocab_size': 11,
        'utterance_len': 6,
        'dim_size': 100,
        'discount_factor': 0.99,
        'learning_rate': 0.001
    }
    GAME_SETTINGS = {
        # 'linguistic_channel': True,
        'batch_size': 2,
        'test_batch_size': 5,
        'episode_num': 5,
        # 'episode_num': 5 * 10 ^ 5
        'item_num': 3,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='wanna see comments?')
    args = parser.parse_args()

    prints = args.__dict__['p'] if args.__dict__['p'] in ['status', 'all'] else None

    SIMULATION_SETTINGS = {
        'prints': prints,
    }

    agents = Agent.create_agents(n=2, **AGENT_SETTINGS)

    game = Game(agents=agents, **GAME_SETTINGS)
    #
    # agents = Agent.create_agents(n=2, settings=SIMULATION_SETTINGS, **AGENT_SETTINGS)
    #
    # game = Game(agents=agents, settings=SIMULATION_SETTINGS, **GAME_SETTINGS)

    game.play()
