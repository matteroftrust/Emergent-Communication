from configparser import SafeConfigParser
from datetime import datetime as dt


def load_settings(config_file='config.ini'):
    config = SafeConfigParser()
    config.read('config.ini')
    print('what is config?', config)
    project_settings = ProjectSettings(**dict(config.items('project_settings')))
    agent_settings = AgentSettings(**dict(config.items('agent_settings')))
    game_settings = GameSettings(**dict(config.items('game_settings')))


    return project_settings, agent_settings, game_settings


class Settings():
    def __str__(self):
        text = 'Settings:\n'
        attributes = self.as_dict.items()
        for key, value in attributes:
            attr_text = '{}: {}\n'.format(key, value)
            text = text + attr_text
        return text[:-1]

    def as_dict(self):
        return self.__dict__


class ProjectSettings(Settings):
    """
    General project settings.
    """

    def __init__(self, prompt='status', validation=True, name=None, acceleration=False):
        self.prompt = prompt
        self.acceleration = acceleration
        self.validation = [True, False][validation in ['False', False]]
        if name:
            self.experiment_name = name
        else:
            now = dt.now()
            self.experiment_name = '{}_experiment'.format(str(now))
        print('experiment nameee', self.experiment_name)

    def __str__(self):
        return 'Project ' + super().__str__()


class AgentSettings(Settings):
    """
    Agent specific settings.
    """
    def __init__(self, lambda_termination=0.05,  # entropy reguralization weight hyperparameter for termination policy
                 lambda_proposal=0.05,  # entropy reguralization weight hyperparameter for proposal policy
                 lambda_utterance=0.001,  # entropy reguralization weight hyperparameter for linguistic utterance policy
                 hidden_state_size=100,
                 vocab_size=11,
                 utterance_len=6,
                 dim_size=100,
                 discount_factor=0.99,
                 learning_rate=0.001,
                 proposal_channel=True,
                 linguistic_channel=False,
                 ):
        self.lambda_termination = lambda_termination
        self.lambda_proposal = lambda_proposal
        self.lambda_utterance = lambda_utterance
        self.hidden_state_size = hidden_state_size
        self.vocab_size = vocab_size
        self.utterance_len = utterance_len
        self.dim_size = dim_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.proposal_channel = [True, False][proposal_channel in ['False', False]]
        self.linguistic_channel = [True, False][linguistic_channel in ['False', False]]

    def __str__(self):
        return 'Agent ' + super().__str__()


class GameSettings(Settings):
    """
    Game specific settings.
    """

    def __init__(self, batch_size=2, test_batch_size=5, episode_num=2, item_num=3, prosocial=False):
        self.batch_size = int(batch_size)
        self.test_batch_size = test_batch_size
        self.episode_num = int(episode_num)
        self.item_num = item_num
        self.prosocial = [True, False][prosocial in ['False', False]]

    def __str__(self):
        return 'Game ' + super().__str__()
