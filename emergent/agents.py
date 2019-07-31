from numpy.random import random_integers
import itertools
import numpy as np

from .game import Action
from .models import CoreLayer, NumberSequenceEncoder, TerminationPolicy, ProposalPolicy, UtterancePolicy


class Agent:

    id_generator = itertools.count()

    def __init__(self, hidden_state_size, vocab_size, dim_size, utterance_len, discount_factor, learning_rate,
                 proposal_channel, linguistic_channel, lambda_termination, lambda_utterance, lambda_proposal):
        self.id = next(self.id_generator)
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        # policies
        self.termination_policy = TerminationPolicy(hidden_state_size, entropy_reg=lambda_termination)
        self.utterance_policy = UtterancePolicy(hidden_state_size=hidden_state_size, is_on=linguistic_channel,
                                                vocab_size=vocab_size, utterance_len=utterance_len, entropy_reg=lambda_utterance)
        self.proposal_policy = ProposalPolicy(hidden_state_size=hidden_state_size, is_on=proposal_channel, entropy_reg=lambda_proposal)

        # NumberSequenceEncoders
        # TODO: input_dim seems to be wrong in here!
        self.context_encoder = NumberSequenceEncoder(input_dim=vocab_size, input_len=6)  # is this correct?
        self.proposal_encoder = NumberSequenceEncoder(input_dim=vocab_size, input_len=3)
        self.utterance_encoder = NumberSequenceEncoder(input_dim=vocab_size, input_len=utterance_len)

        # feedforward layer that takes (h_c, h_m, h_p) and returns hidden_state
        self.core_layer = CoreLayer()

    def __str__(self):
        return 'agent {}'.format(self.id)

    @classmethod
    def create_agents(self, n, *args, **kwargs):
        agents = [Agent(*args, **kwargs) for _ in range(n)]
        return agents

    def generate_util_fun(self):
        """
        Generate utility function which specifies rewards for each item.
        """
        while True:
            out = random_integers(0, 10, 3)
            if list(out) != [0, 0, 0]:
                self.utilities = out
                return out

    def propose(self, h_c, utterance, proposal, termination_true=False, test=False, **kwargs):
        h_m, h_p = self.utterance_encoder(utterance), self.proposal_encoder(proposal)
        input = np.concatenate([h_c, h_m, h_p])
        input = input.reshape(1, -1)
        hidden_state = self.core_layer(input)

        if termination_true:
            termination = False
        else:
            termination = self.termination_policy(hidden_state, test=test)
        # if termination:
        #     action = Action(terminate=termination, utterance=utterance, proposal=proposal, id=self.id)
        # TODO: if atermination == True then we dont need utterance and proposal but what about training?
        utterance = self.utterance_policy(hidden_state)  # should test also be passed here?
        proposal = self.proposal_policy(hidden_state, **kwargs)  # should test also be passed here?
        hidden_state = hidden_state.reshape(-1)

        action = Action(terminate=termination, utterance=utterance, proposal=proposal, id=self.id)

        return action, hidden_state
