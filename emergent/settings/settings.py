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

    def __init__(self, prompt=None):
        self.prompt = prompt

    def __str__(self):
        return 'Project ' + super().__str__()

    @classmethod
    def default(self):
        return ProjectSettings(prompt='status')


class AgentSettings(Settings):
    """
    Agent specific settings.
    """
    def __init__(self, lambda_termination, lambda_proposal, lambda_utterance, hidden_state_size, vocab_size,
                 utterance_len, dim_size, discount_factor, learning_rate):
        self.lambda_termination = lambda_termination
        self.lambda_proposal = lambda_proposal
        self.lambda_utterance = lambda_utterance
        self.hidden_state_size = hidden_state_size
        self.vocab_size = vocab_size
        self.utterance_len = utterance_len
        self.dim_size = dim_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

    def __str__(self):
        return 'Agent ' + super().__str__()


class GameSettings(Settings):
    """
    Game specific settings.
    """

    def __init__(self, batch_size, test_batch_size, episode_num, item_num):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.episode_num = episode_num
        self.item_num = item_num

    def __str__(self):
        return 'Game ' + super().__str__()
