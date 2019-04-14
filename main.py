from game import Game
from agents import Agent

if __name__ == '__main__':

    AGENT_SETTINGS = {
        'lambda_term': 0.05,  # entropy reguralization weight hyperparameter for termination policy
        'lambda_prop': 0.05,  # entropy reguralization weight hyperparameter for proposal policy
        'lambda_utt': 0.001,  # entropy reguralization weight hyperparameter for linguistic utterance policy
        'hidden_state_size': 100,
        'vocab_size': 10,
        'dim_size': 100
    }

    GAME_SETTINGS = {
        # 'linguistic_channel': True,
        'batch_size': 128,
        'episode_num': 5,
        # 'episode_num': 5 * 10 ^ 5
        'item_num': 3,
        'utter_len': 10,
    }

    agents = Agent.create_agents(n=2, **AGENT_SETTINGS)

    game = Game(agents=agents, **GAME_SETTINGS)

    game.play()
