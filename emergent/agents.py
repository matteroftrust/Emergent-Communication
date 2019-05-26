from numpy.random import random_integers
import itertools
import numpy as np

from .game import Action
from .settings import load_settings
from .utils import print_all, print_status, validation

from keras.activations import sigmoid
from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential


project_settings, agent_settings, game_settings = load_settings()


class NumberSequenceEncoder:
    def __init__(self, input_dim, output_dim, hidden_state_size=100):
        """
        item_dim is a number of different values that can occur as unput. I.e. for utterance input_dim=vocab_size.
        """
        self.model = Sequential([
            Embedding(input_dim=input_dim, output_dim=output_dim),
            LSTM(hidden_state_size)
        ])

    def __call__(self, input):
        return self.encode(input)

    def encode(self, input):
        return self.model.predict(input)


class Policy:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(*args, **kwargs):
        pass

    @validation
    def input_is_valid(self, input):
        expected_shape = (1, 100, 1)
        is_valid = input.shape == expected_shape
        msg = '{} input invalid. Expected: {} received: {}'.format(self.__class__.__name__, expected_shape, input.shape)
        return is_valid, msg

    @validation
    def output_is_valid(self, output, expected_shape):
        is_valid = output.shape == expected_shape
        msg = '{} output invalid. Expected: {} received: {}\noutput: {}'.format(self.__class__.__name__, expected_shape, output.shape, output)
        return is_valid, msg


class TerminationPolicy(Policy):
    """
    This is a binary decision, and we parametrise πterm as a single feedforward layer,
    with the hidden state as input, followed by a sigmoid function, to represent the probability of termination.
    """
    def __init__(self, hidden_state_size, entropy_reg=0.05):
        # single feedforward layer with sigmoid function
        self.model = Sequential([
            Dense(1, input_shape=(hidden_state_size, 1)),
            # sigmoid()
            Activation('sigmoid')
        ])
        # takes (batch_size, hidden_state_size) vectors as input

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',  # TODO these are random, needs to be checked
                           metrics=['accuracy'])

    @validation
    def output_is_valid(self, output):
        is_valid = type(output) in [bool, np.bool, np.bool_]
        msg = '{} output invalid. Expected: {} received: {}'.format(self.__class__.__name__, type(output), 'boolean')
        return is_valid, msg

    def forward(self, hidden_state):
        self.input_is_valid(hidden_state)
        confidence = self.model.predict(hidden_state) # this should return just a value or an array of values for a batch input
        print_all('TerminationPolicy output dim: {}'.format(confidence.shape))
        confidence = np.mean(confidence)  # TODO this is completely wrong, I know
        out = np.random.random() <= confidence # we sample with probability, TOOD should find something more elegant
        self.output_is_valid(out)
        return out

    def train(self):
        pass


class UtterancePolicy(Policy):
    """
    This is parametrised by an LSTM, which takes the hidden state of the agent as the initial hidden state.
    For the first timestep, a dummy symbol is fed in as input;
    subsequently, the model prediction from the previous timestep is fed in as input at the next timestep,
    in order to predict the next symbol
    """
    def __init__(self, hidden_state_size, entropy_reg=0.001):
        self.model = Sequential([
            LSTM(hidden_state_size, input_shape=(100, 1))
        ])
        self.model.compile(optimizer='adam',
                           loss='mse',  # TODO these are random, needs to be checked
                           metrics=['accuracy'])

    @property
    def dummy(self):
        return np.zeros(6)

    def forward(self, hidden_state, utterance_channel=False):
        if not utterance_channel:
            return self.dummy
        self.input_is_valid(hidden_state)
        utterance = self.model.predict(hidden_state)
        self.output_is_valid(utterance, (6))
        return utterance

    def train(self):
        pass


class ProposalPolicy(Policy):
    """
    This is parametrised by 3 separate feedforward neural networks, one for each item type,
    which each take as input ht and output a distribution over {0...5} indicating the proposal for that item
    """
    def __init__(self, hidden_state_size, item_num=3, entropy_reg=0.05):
        self.item_num = item_num
        self.models = []
        for _ in range(self.item_num):
            model = Sequential([
                LSTM(1, input_shape=(hidden_state_size, 1))
            ])
            model.compile(optimizer='adam',
                          loss='mse',  # TODO these are random, needs to be checked
                          metrics=['accuracy'])
            self.models.append(model)

    def forward(self, hidden_state):
        self.input_is_valid(hidden_state)
        proposal = []
        for i in range(self.item_num):
            single_proposal = self.models[i].predict(hidden_state)
            single_proposal = int(single_proposal)
            proposal.append(single_proposal)
        out = np.array(proposal)
        self.output_is_valid(out, (3,))
        return out

    def train(self, ):
        pass


class Agent:

    id_generator = itertools.count()

    def __init__(self, lambda_termination, lambda_proposal,
                 lambda_utterance, hidden_state_size, vocab_size,
                 dim_size, utterance_len, discount_factor, learning_rate,
                 utterance_channel):
        self.id = next(self.id_generator)
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.lambda_termination = lambda_termination
        self.lambda_proposal = lambda_proposal
        self.lambda_utterance = lambda_utterance

        # policies
        self.termination_policy = TerminationPolicy(hidden_state_size)
        self.utterance_policy = UtterancePolicy(hidden_state_size)
        self.proposal_policy = ProposalPolicy(hidden_state_size)

        self.utterance_channel = utterance_channel
        self.utterance_len = utterance_len
        self.vocab_size = vocab_size

        # NumberSequenceEncoders
        self.context_encoder = NumberSequenceEncoder(input_dim=self.vocab_size, output_dim=hidden_state_size)  # is this correct?
        # self.proposal_encoder = NumberSequenceEncoder(input_dim=6, output_dim=hidden_state_size)
        self.utterance_encoder = NumberSequenceEncoder(input_dim=self.utterance_len, output_dim=hidden_state_size)

        # feedforward layer that takes (h_c, h_m, h_p) and returns hidden_state
        self.core_layer_model = Sequential([
            Dense(100, input_shape=(1500,), name="{}_dense".format(self.id)),
            Activation('relu'),
        ])
        self.core_layer = self.core_layer_model.predict

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

    def propose(self, context, utterance, proposal):
        print_all('# Proposal by {} previous proposal {}'.format(self.id, proposal))
        h_c, h_m, h_p = self.context_encoder(context), self.utterance_encoder(utterance), self.context_encoder(proposal)
        input = np.concatenate([h_c, h_m, h_p])
        print_all('hidden state original: {}'.format(input.shape))
        input = np.reshape(input, (1, 1500))
        print_all('hidden state : {}'.format(input.shape))
        hidden_state = self.core_layer(input)
        hidden_state = np.expand_dims(hidden_state, axis=2)
        print_all('hidden state after: {}'.format(hidden_state.shape))

        # hidden_state = np.expand_dims(hidden_state, axis=2)
        termination = self.termination_policy(hidden_state)
        utterance = self.utterance_policy(hidden_state)
        proposal = self.proposal_policy(hidden_state)

        action = Action(terminate=termination, utterance=utterance, proposal=proposal, id=self.id)

        return action


    def reward(self, reward):
        pass

    # Discounting rewards collected in an episode.
    # e.g discount_factor = 0.99 then [1, 1, 1, 1] -> [3.94, 2.97, 1.99, 1.0]
    # line 5 https://github.com/breeko/Simple-Reinforcement-Learning-with-Tensorflow/blob/master/Part%202%20-%20Policy-based%20Agents%20with%20Keras.ipynb
    # line 61 https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/3-reinforce/cartpole_reinforce.py
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in range(len(rewards) - 1, -1, -1):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
