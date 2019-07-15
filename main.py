from configparser import SafeConfigParser
import argparse
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    # we might want to use it for controlling speed of the script
    # https://docs.python.org/3.6/library/profile.html

    if os.path.isfile('config.ini'):
        print('removinnn')
        os.remove('config.ini')

    for dir in ['results', 'figs']:
        if not os.path.isdir(dir):
            os.makedirs(dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='prompt', help='wanna see comments?')
    parser.add_argument('-v', action='store_true', dest='validation', help='data validation?')
    parser.add_argument('--batch_size', dest='batch_size', type=int)
    parser.add_argument('--test_batch_size', dest='test_batch_size', type=int)
    parser.add_argument('--episode_num', dest='episode_num', type=int)
    parser.add_argument('--acceleration', dest='acceleration')
    parser.add_argument('--channels', dest='channels')
    parser.add_argument('--prosocial', dest='prosocial')
    parser.set_defaults(validation=False, prompt='status', batch_size=2, test_batch_size=2, episode_num=2,
                        acceleration=False, channels='proposal', prosocial=False)
    args = parser.parse_args()

    prompt = args.__dict__['prompt']
    validation = args.__dict__['validation']
    batch_size = args.__dict__['batch_size']
    test_batch_size = args.__dict__['test_batch_size']
    episode_num = args.__dict__['episode_num']
    acceleration = args.__dict__['acceleration']
    channels = args.__dict__['channels'].split(',')
    prosocial = args.__dict__['prosocial']

    config = SafeConfigParser()
    config.read('config.ini')
    config.add_section('project_settings')
    config.set('project_settings', 'prompt', prompt)
    config.set('project_settings', 'validation', str(validation))
    config.set('project_settings', 'acceleration', str(acceleration))

    config.add_section('game_settings')
    config.set('game_settings', 'batch_size', str(batch_size))
    config.set('game_settings', 'test_batch_size', str(test_batch_size))
    config.set('game_settings', 'episode_num', str(episode_num))
    config.set('game_settings', 'prosocial', str(prosocial))

    config.add_section('agent_settings')
    config.set('agent_settings', 'linguistic_channel', str('linguistic' in channels))
    config.set('agent_settings', 'proposal_channel', str('proposal' in channels))

    with open('config.ini', 'a') as f:
        config.write(f)

    # emergent module has to be imported after config.init file is created.
    import emergent
    from emergent.utils import print_status, print_all

    # project_settings = emergent.settings.ProjectSettings(
    #     prompt=prompt,
    #     validation=validation,
    #     acceleration=acceleration
    # )

    project_settings, agent_settings, game_settings = emergent.settings.load_settings()

    for settings in [project_settings, agent_settings, game_settings]:
        print(settings)

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
