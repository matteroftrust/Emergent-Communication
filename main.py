from configparser import SafeConfigParser
import argparse
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':

    # we might want to use it for controlling speed of the script
    # https://docs.python.org/3.6/library/profile.html

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='prompt', help='wanna see comments?')
    parser.add_argument('-v', action='store_true', dest='validation', help='data validation?')
    parser.add_argument('--batch_size', dest='batch_size', type=int)
    parser.add_argument('--test_batch_size', dest='test_batch_size', type=int)
    parser.add_argument('--episode_num', dest='episode_num', type=int)
    parser.set_defaults(validation=False, prompt='status', batch_size=2, test_batch_size=2, episode_num=2)
    args = parser.parse_args()

    prompt = args.__dict__['prompt']
    validation = args.__dict__['validation']
    batch_size = args.__dict__['batch_size']
    test_batch_size = args.__dict__['test_batch_size']
    episode_num = args.__dict__['episode_num']


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
    from emergent.utils import print_status, print_all

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

    game_settings = emergent.settings.GameSettings(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        episode_num=episode_num,
    )
        # 'linguistic_channel': True,
        # # 'episode_num': 5 * 10 ^ 5
        # item_num=3

    print_status('### Agents initialization. ###\n')
    agents = emergent.Agent.create_agents(n=2, **agent_settings.as_dict())
    print_all('agents summary')
    for agent in agents:
        print('\nsummary {}'.format(agent.id))
        print(agent.core_layer_model.summary())

    print_status('\n### Game initialization. ###\n')
    game = emergent.Game(agents=agents, **game_settings.as_dict())

    print_status('\n### Starting experiment. ###\n')
    game.play()
    print_status('\n### ### Done. ### ###\n')
