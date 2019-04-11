from game import Game
from agents import Agent

if __name__ == '__main__':

    LAMBDA_TERM = 0.05  # entropy reguralization weight hyperparameter for termination policy
    LAMBDA_PROP = 0.05  # entropy reguralization weight hyperparameter for proposal policy
    LAMBDA_UTT = 0.001  # entropy reguralization weight hyperparameter for linguistic utterance policy

    END = 10

    AGENT_SETTINGS = {
        'vocab_size': 11,
        'dim_size': 100
    }

    GAME_SETTINGS = {
        'linguistic_channel': True,
    }

    agents = Agent.create_agents(2, lambda_term=LAMBDA_TERM, lambda_prop=LAMBDA_PROP, lambda_utt=LAMBDA_UTT, hidden_state_size=100)

    game = Game(end=END, agents=agents, settings=GAME_SETTINGS)

    game.play()
