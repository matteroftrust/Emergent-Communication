import argparse
import os
from configparser import SafeConfigParser

import emergent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Config:
    def __init__(*kwargs):
        pass


if __name__ == '__main__':

    agent_settings = emergent.settings.AgentSettings(
        lambda_termination=0.05,  # entropy reguralization weight hyperparameter for termination policy
        lambda_proposal=0.05,  # entropy reguralization weight hyperparameter for proposal policy
        lambda_utterance=0.001,  # entropy reguralization weight hyperparameter for linguistic utterance policy
        hidden_state_size=100,
        vocab_size=11,
        utterance_len=6,
        dim_size=100,
        discount_factor=0.99,
        learning_rate=0.001
    )

    game_settings = emergent.settings.GameSettings(
        # 'linguistic_channel': True,
        batch_size=2,
        test_batch_size=5,
        episode_num=5,
        # 'episode_num': 5 * 10 ^ 5
        item_num=3
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='prompt', help='wanna see comments?')
    parser.add_argument('--v', action='store_true', dest='validation', help='data validation?')
    parser.set_defaults(validation=False, prompt='status')
    args = parser.parse_args()

    prompt = args.__dict__['prompt']
    validation = args.__dict__['validation']

    project_settings = emergent.settings.ProjectSettings(
        prompt=prompt,
        validation=validation
    )
    print('validation {} prompt {}'.format(validation, prompt))
    try:
        os.remove('config.ini')
    except:
        pass
    config = SafeConfigParser()
    config.read('config.ini')
    config.add_section('project_settings')
    config.set('project_settings', 'prompt', prompt)
    config.set('project_settings', 'validation', str(validation))
    # config.set('main', 'key2', 'value2')
    # config.set('main', 'key3', 'value3')

    with open('config.ini', 'w') as f:
        config.write(f)

    agents = emergent.Agent.create_agents(n=2, **agent_settings.as_dict())

    game = emergent.Game(agents=agents, **game_settings.as_dict())
    #
    # agents = Agent.create_agents(n=2, settings=SIMULATION_SETTINGS, **AGENT_SETTINGS)
    #
    # game = Game(agents=agents, settings=SIMULATION_SETTINGS, **GAME_SETTINGS)

    game.play()

    # remove config file

    os.remove('config.ini')
