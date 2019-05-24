from configparser import SafeConfigParser
import argparse
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='prompt', help='wanna see comments?')
    parser.add_argument('-v', action='store_true', dest='validation', help='data validation?')
    parser.set_defaults(validation=False, prompt='status')
    args = parser.parse_args()

    prompt = args.__dict__['prompt']
    validation = args.__dict__['validation']


    try:
        os.remove('config.ini')
    except:
        pass

    config = SafeConfigParser()
    config.read('config.ini')
    config.add_section('project_settings')
    config.set('project_settings', 'prompt', prompt)
    config.set('project_settings', 'validation', str(validation))

    with open('config.ini', 'a') as f:
        config.write(f)

    # emergent module has to be imported after config.init file is created.
    import emergent
    from emergent.utils import print_status

    project_settings = emergent.settings.ProjectSettings(
        prompt=prompt,
        validation=validation
    )

    agent_settings = emergent.settings.AgentSettings(
        lambda_termination=0.05,  # entropy reguralization weight hyperparameter for termination policy
        lambda_proposal=0.05,  # entropy reguralization weight hyperparameter for proposal policy
        lambda_utterance=0.001,  # entropy reguralization weight hyperparameter for linguistic utterance policy
        hidden_state_size=100,
        vocab_size=11,
        utterance_len=6,
        dim_size=100,
        discount_factor=0.99,
        learning_rate=0.001,
        utterance_channel=False
    )

    game_settings = emergent.settings.GameSettings()
        # 'linguistic_channel': True,
        # batch_size=2,
        # test_batch_size=5,
        # episode_num=5,
        # # 'episode_num': 5 * 10 ^ 5
        # item_num=3

    print_status('### Agents initialization. ###\n')
    agents = emergent.Agent.create_agents(n=2, **agent_settings.as_dict())

    print_status('\n### Game initialization. ###\n')
    game = emergent.Game(agents=agents, **game_settings.as_dict())

    print_status('\n### Starting experiment. ###\n')
    game.play()
    print_status('\n### ### Done. ### ###\n')
