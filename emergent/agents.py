from numpy.random import random_integers
import itertools
import numpy as np

from .game import Action
from .settings import load_settings
from .utils import print_all, print_status, validation, convert_to_sparse

from keras.activations import sigmoid
from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD


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
        expected_shape = (100,)
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
            Dense(1, input_shape=(hidden_state_size,)),
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
        msg = '{} output invalid. Expected: {} received: {}'.format(self.__class__.__name__, 'boolean', type(output))
        return is_valid, msg

    # @validation
    # def train_batch_is_valid(self, x, y, sample_weight):
    #     msg = ''
    #     if x.shape[0] != 100:
    #         msg += 'Invalid shape of x. Expected {} received (100,)\n'.format(x.shape)
    #     if y.shape[0] != 1:
    #         msg += 'Invalid shape of x. Expected {} received (1,)\n'.format(x.shape)
    #     if x.shape[0] != 100:
    #         msg += 'Invalid shape of x. Expected {} received (100,)\n'.format(x.shape)

    def forward(self, hidden_state):
        self.input_is_valid(hidden_state)
        hidden_state = np.expand_dims(hidden_state, 0)
        confidence = self.model.predict(hidden_state)[0][0]
        if np.isnan(confidence):
            print('whattheeeoo termination forward returns nan!')
            print(self.model.get_weights())
        # out = np.random.random() <= confidence  # we sample with probability, TOOD should find something more elegant
        out = np.random.choice([True, False], p=[confidence, 1 - confidence])
        self.output_is_valid(out)
        return out

    def train(self, x, y, sample_weight):
        out = self.model.train_on_batch(x, y, sample_weight=sample_weight)
        if np.isnan(self.model.get_weights()[0].any()):
            print('this traininng went wronngggg!!!')
        return out


class UtterancePolicy(Policy):
    """
    This is parametrised by an LSTM, which takes the hidden state of the agent as the initial hidden state.
    For the first timestep, a dummy symbol is fed in as input;
    subsequently, the model prediction from the previous timestep is fed in as input at the next timestep,
    in order to predict the next symbol
    """
    def __init__(self, hidden_state_size, utterance_policy=False, entropy_reg=0.001):
        if not utterance_policy:
            # should return some dummy policy
            pass
        self.model = Sequential([
            LSTM(hidden_state_size, input_shape=(hidden_state_size, 1))
        ])
        self.model.compile(optimizer='adam',
                           loss='mse',  # TODO these are random, needs to be checked
                           metrics=['accuracy'])

    @property
    def dummy(self):
        return np.zeros(6)

    def forward(self, hidden_state, utterance_channel=False):
        if not utterance_channel:
            utterance = self.dummy
        else:
            hidden_state = np.expand_dims(np.expand_dims(hidden_state, 0), 2)
            self.input_is_valid(hidden_state)
            utterance = self.model.predict(hidden_state)
        self.output_is_valid(utterance, (6,))
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
                Dense(6, input_shape=(hidden_state_size,)),
                Activation('softmax')
            ])
            # model.compile(optimizer='adam',
            #               loss='mse',  # TODO these are random, needs to be checked
            #               metrics=['accuracy']

            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.5)  # SGD?
            model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])

            # model.compile = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

            self.models.append(model)

    def forward(self, hidden_state):
        self.input_is_valid(hidden_state)
        hidden_state = np.expand_dims(hidden_state, 0)
        proposal = []
        for i in range(self.item_num):
            distribution = self.models[i].predict(hidden_state)
            single_proposal = np.random.choice(np.arange(6), p=distribution[0])
            proposal.append(single_proposal)
        out = np.array(proposal)
        self.output_is_valid(out, (3,))
        return out

    def train(self, x, y, sample_weight):
        # print('train proposal policy shape x {} y {} sam {}'.format(x.shape, y.shape, sample_weight.shape))

        # x = np.expand_dims(x, 2)
        for i in range(self.item_num):
            self.models[i].train_on_batch(x, convert_to_sparse(y[:, i], 6), sample_weight=sample_weight)

# class HiddenStateNetwork(Policy):
#     """
#     Formally, its not a policy be why wouldnt we inherit from Policy class if we can.
#     """
#     def __init__(self, vocab_size, utterance_len, hidden_state_size=100):
#         self.context_encoder = NumberSequenceEncoder(input_dim=self.vocab_size, output_dim=hidden_state_size)  # is this correct?
#         # self.proposal_encoder = NumberSequenceEncoder(input_dim=6, output_dim=hidden_state_size)
#         self.utterance_encoder = NumberSequenceEncoder(input_dim=self.utterance_len, output_dim=hidden_state_size)
#
#         self.core_layer_model = Sequential([
#             Dense(hidden_state_size, input_shape=(1500,), name="{}_dense".format(self.id)),
#             Activation('relu'),
#         ])
#         self.core_layer = self.core_layer_model.predict


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
        self.utterance_channel = utterance_channel
        self.utterance_len = utterance_len
        self.vocab_size = vocab_size

        self.termination_policy = TerminationPolicy(hidden_state_size)
        self.utterance_policy = UtterancePolicy(hidden_state_size, utterance_channel)
        self.proposal_policy = ProposalPolicy(hidden_state_size)

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
        hidden_state = np.reshape(hidden_state, (100,))
        print_all('hidden state after: {}'.format(hidden_state.shape))

        # hidden_state = np.expand_dims(hidden_state, axis=2)
        termination = self.termination_policy(hidden_state)
        utterance = self.utterance_policy(hidden_state)
        proposal = self.proposal_policy(hidden_state)
        hidden_state = np.reshape(hidden_state, 100)  # TODO should be fixed before in models

        action = Action(terminate=termination, utterance=utterance, proposal=proposal, id=self.id)

        return action, hidden_state

    def train(self, x, rewards, y_termination, y_proposal, y_utterance=None):
        self.termination_policy.train(x, y_termination, rewards)
        self.proposal_policy.train(x, y_proposal, rewards)
        if y_utterance:
            self.utterance_policy.train(x, y_utterance, rewards)

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
