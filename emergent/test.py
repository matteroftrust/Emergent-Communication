from .agents import Agent, TerminationPolicy, ProposalPolicy, UtterancePolicy
from .game import Game


class Response:
    def __init__(self, response):
        self.response = response

    def forward(self, *args, **kwargs):
        return self.response


class TestPolicy:
    def train(self, *args, **kwargs):
        pass


class StaticTestTerminationPolicy(TerminationPolicy, TestPolicy):
    def __init__(self, hidden_state_size, response):
        self.model = Response(response)


class StaticTestProposalPolicy(ProposalPolicy, TestPolicy):
    def __init__(self, response, is_on=True, hidden_state_size=100, item_num=3, **kwargs):
        self.is_on = is_on
        self.item_num = item_num
        if is_on:
            self.model = Response(response)


class StaticTestUtterancePolicy(UtterancePolicy, TestPolicy):
    def __init__(self, response, is_on=3, vocab_size=11, utterance_len=6, *args, **kwargs):
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
        super(Agent, self).__init__(*args, **kwargs)
        # self. = StaticTestTerminationPolicy(kwargs)
        self.proposal_policy = StaticTestProposalPolicy(self, prop_response, *args, **kwargs)
        self.utterance_policy = StaticTestUtterancePolicy(self, utter_response, *args, **kwargs)

        @classmethod
        def create_agents(self, n, *args, **kwargs):
            agents = [Agent(*args, **kwargs) for _ in range(n)]
            return agents


class TestStaticGame(Game):
    pass
