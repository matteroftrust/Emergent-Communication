from numpy.random import random_integers
import itertools
import numpy as np

from .game import Action
from .utils import print_all, print_status, validation, convert_to_sparse

from keras import Input, regularizers
from keras.layers import Dense, Activation, LSTM
from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model
# from keras.optimizers import SGD


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
            Dense(1, input_shape=(hidden_state_size,),
                  # kernel_initializer='random_uniform',  # TODO or maybe random_normal
                  kernel_initializer='random_normal',  # TODO or maybe random_normal
                  activity_regularizer=regularizers.l1(entropy_reg)
                  ),
            Activation('sigmoid')
        ])

        # Accuracy is not the right measure for your model's performance. What you are trying to do here is more of a
        # regression task than a classification task. The same can be seen from your loss function, you are using
        # 'mean_squared_error' rather than something like 'categorical_crossentropy'.
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',  # TODO these are random, needs to be checked
                           metrics=['accuracy'])
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.5)  # SGD?
        # # self.model.compile(loss='categorical_crossentropy',
        # self.model.compile(loss='mean_squared_error',
        #
        #                    optimizer=sgd,
        #                    metrics=['accuracy'])

    @validation
    def output_is_valid(self, output):
        is_valid = type(output) in [bool, np.bool, np.bool_]
        msg = '{} output invalid. Expected: {} received: {}'.format(self.__class__.__name__, 'boolean', type(output))
        return is_valid, msg

    def forward(self, hidden_state, test=False):
        self.input_is_valid(hidden_state)
        hidden_state = np.expand_dims(hidden_state, 0)
        confidence = self.model.predict(hidden_state)[0][0]
        if test:
            out = [True, False][confidence < 0.5]
        else:
            out = np.random.choice([True, False], p=[confidence, 1 - confidence])
        self.output_is_valid(out)
        return out

    def train(self, x, y, sample_weight):
        out = self.model.train_on_batch(x, y, sample_weight=sample_weight)
        return out


class UtterancePolicy(Policy):
    """
    This is parametrised by an LSTM, which takes the hidden state of the agent as the initial hidden state.
    For the first timestep, a dummy symbol is fed in as input;
    subsequently, the model prediction from the previous timestep is fed in as input at the next timestep,
    in order to predict the next symbol

    The symbol vocabulary size was 11, and the agents were allowed to generate utterances of up to length 6.
    """
    def __init__(self, hidden_state_size, is_on=True, vocab_size=11, utterance_len=6, entropy_reg=0.001):
        """
        lambda is an entropy regularization term
        """
        self.is_on = is_on
        self.utterance_len = utterance_len
        self.vocab = list(range(vocab_size))
        self.vocab_size = vocab_size

        if self.is_on:
            inputs = Input(batch_shape=(1, 1, 1), name='utter_input')
            lstm1 = LSTM(100, stateful=True, name='utter_lstm',
                         activity_regularizer=regularizers.l1(entropy_reg))(inputs)
            dense = Dense(vocab_size, activation='softmax', name='utter_dense')(lstm1)
            model = Model(inputs=inputs, outputs=[dense])
            model.compile(optimizer='adam',
                          loss='mse',
                          # TODO might be cool to use the one below (requires different shape in training)
                          # loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'],
                          # sample_weight_mode="temporal"
                          )
            self.model = model

    @property
    def dummy(self):
        return np.zeros(6)

    @property
    def dummy_symbol(self):
        return np.zeros((1, 1, 1))

    def forward(self, hidden_state):
        if not self.is_on:
            utterance = self.dummy
        else:
            # hidden_state = np.expand_dims(np.expand_dims(hidden_state, 0), 2)
            # self.input_is_valid(hidden_state)
            if hidden_state is not None:  # if hidden state is passed then we set is as a new LSTM state
                self.model.layers[1].states[0] = hidden_state
            utterance = [self.vocab[self.model.predict(self.dummy_symbol).argmax()]]
            for i in range(self.utterance_len - 1):
                last_symbol = utterance[-1] * np.ones((1, 1, 1))
                arg_max = self.model.predict(last_symbol).argmax()
                utterance.append(self.vocab[arg_max])
            print_all('this is utterance!!', utterance)

        self.output_is_valid(utterance, (6,))
        return utterance

    def train(self, x, y, sample_weight):
        if self.is_on:
            for xx, yy, ssww in zip(x, y, sample_weight):
                inputs = [self.dummy_symbol] + yy[-1]
                ssww = np.array([ssww])
                for xxx, yyy in zip(inputs, yy):
                    # print('pewnie kurwa nie dziala xysw', xx.shape, yy.shape, ssww.shape)
                    ar = np.zeros(self.vocab_size)
                    ar[yyy] = 1
                    yyy = ar.reshape(1, self.vocab_size)
                    self.model.train_on_batch(xxx, yyy, sample_weight=ssww)


class ProposalPolicy(Policy):
    """
    This is parametrised by 3 separate feedforward neural networks, one for each item type,
    which each take as input ht and output a distribution over {0...5} indicating the proposal for that item
    """
    def __init__(self, is_on, hidden_state_size=100, item_num=3, entropy_reg=0.05):
        self.is_on = is_on
        self.item_num = item_num
        if self.is_on:
            self.models = []
            for _ in range(self.item_num):
                model = Sequential([
                    Dense(6, input_shape=(hidden_state_size,),
                          activity_regularizer=regularizers.l1(entropy_reg)),
                    Activation('softmax')
                ])
                model.compile(optimizer='adam',
                              loss='mse',  # TODO these are random, needs to be checked
                              metrics=['accuracy'])

                self.models.append(model)

    @property
    def dummy(self):
        return np.zeros(self.item_num)

    def forward(self, hidden_state):
        self.input_is_valid(hidden_state)
        if not self.is_on:
            return self.dummy
        hidden_state = np.expand_dims(hidden_state, 0)
        proposal = []
        for i in range(self.item_num):
            distribution = self.models[i].predict(hidden_state)[0]
            single_proposal = np.random.choice(np.arange(6), p=distribution)
            proposal.append(single_proposal)
        out = np.array(proposal)
        self.output_is_valid(out, (3,))
        return out

    def train(self, x, y, sample_weight):
        if self.is_on:
            for i in range(self.item_num):
                y = convert_to_sparse(y[:, i], 6)
                # print('whats there for training in proposal policy??? \n', x.shape, y.shape, sample_weight.shape)
                self.models[i].train_on_batch(x, y, sample_weight=sample_weight)

    def get_weights(self):
        out = [model.get_weights() for model in self.models]
        return out


class Agent:

    id_generator = itertools.count()

    def __init__(self, lambda_termination, lambda_proposal,
                 lambda_utterance, hidden_state_size, vocab_size,
                 dim_size, utterance_len, discount_factor, learning_rate,
                 proposal_channel, linguistic_channel):
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
        self.context_encoder = NumberSequenceEncoder(input_dim=vocab_size, output_dim=hidden_state_size)  # is this correct?
        # self.proposal_encoder = NumberSequenceEncoder(input_dim=6, output_dim=hidden_state_size)
        self.utterance_encoder = NumberSequenceEncoder(input_dim=vocab_size, output_dim=hidden_state_size)

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

    def propose(self, context, utterance, proposal, test=False):
        h_c, h_m, h_p = self.context_encoder(context), self.utterance_encoder(utterance), self.context_encoder(proposal)
        input = np.concatenate([h_c, h_m, h_p])
        input = np.reshape(input, (1, 1500))
        hidden_state = self.core_layer(input)
        hidden_state = np.reshape(hidden_state, (100,))

        termination = self.termination_policy(hidden_state, test=test)
        utterance = self.utterance_policy(hidden_state)  # should test also be passed here?
        proposal = self.proposal_policy(hidden_state)  # should test also be passed here?
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
