from .agents import Agent
from .game import Game
from .models import TerminationPolicy, ProposalPolicy, UtterancePolicy, NumberSequenceEncoder, CoreLayer

from keras.layers import Dense, Activation
from keras.models import Sequential


class Response:
    def __init__(self, response):
        self.response = response

    def forward(self, *args, **kwargs):
        return self.response

    def train_on_batch(self, *args, **kwargs):
        pass


class TestPolicy:

    def forward(self, *args, **kwargs):
        return self.model.forward()

    def train_on_batch(self, *args, **kwargs):
        pass


class StaticTestTerminationPolicy(TerminationPolicy, TestPolicy):
    def __init__(self, response):
        self.model = Response(response)
        self.forward = self.model.forward


class StaticTestProposalPolicy(ProposalPolicy, TestPolicy):
    def __init__(self, response, is_on=True, item_num=3, **kwargs):
        self.is_on = is_on
        self.item_num = item_num
        if is_on:
            self.model = Response(response)
        self.forward = self.model.forward

    def train(self, *args, **kwargs):
        pass


class StaticTestUtterancePolicy(UtterancePolicy, TestPolicy):
    def __init__(self, response, is_on=False, vocab_size=11, utterance_len=6, *args, **kwargs):
        self.is_on = is_on
        self.utterance_len = utterance_len
        self.vocab = list(range(vocab_size))
        self.vocab_size = vocab_size

        if self.is_on:
            self.model = Response(response)


class TestStaticAgent(Agent):
    def __init__(self, term_response, prop_response, utter_response, *args, **kwargs):
        print(args)
        print(kwargs)

        if 'id' in kwargs:
            self.id = kwargs['id']
        else:
            self.id = next(self.id_generator)

        discount_factor = 0.99
        learning_rate = 0.001

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.termination_policy = StaticTestTerminationPolicy(response=term_response)
        self.proposal_policy = StaticTestProposalPolicy(response=prop_response, is_on=True, **kwargs)
        self.utterance_policy = StaticTestUtterancePolicy(response=utter_response, is_on=False, **kwargs)

        self.context_encoder = NumberSequenceEncoder(input_dim=11, input_len=6)
        self.proposal_encoder = NumberSequenceEncoder(input_dim=6, input_len=3)
        self.utterance_encoder = NumberSequenceEncoder(input_dim=11, input_len=6)

        self.core_layer = CoreLayer()

        @classmethod
        def create_agents(self, n, *args, **kwargs):
            agents = [TestStaticAgent(*args, **kwargs) for _ in range(n)]
            return agents


class TestStaticGame(Game):
    pass
