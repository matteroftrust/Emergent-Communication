import numpy as np

from .utils import validation, convert_to_sparse

from tensorflow.python.keras import Input, regularizers, optimizers
from tensorflow.python.keras.layers import Dense, Activation, LSTM, Flatten
from tensorflow.python.keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM
from tensorflow.python.keras.models import Sequential, Model
# from keras.optimizers import SGD
from tensorflow.python.keras.utils import to_categorical


class CoreLayer:
    def __init__(self):
        self.model = Sequential([
            Dense(100, input_shape=(300,), name="dense"),
            Activation('relu'),
        ])

    def __call__(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


class NumberSequenceEncoder:
    def __init__(self, input_dim, input_len, hidden_state_size=100):
        """
        item_dim is a number of different values that can occur as unput. I.e. for utterance input_dim=vocab_size.
        """
        # self.model = Sequential([
        #     Flatten(input_dim=input_dim, output_dim=output_dim),
        #     Embedding(),
        #     LSTM(hidden_state_size)
        # ])

        self.model = Sequential([
            Embedding(input_dim=input_dim, output_dim=hidden_state_size, input_length=input_len),
        ])
        self.lstm = Sequential([LSTM(input_shape=(1, input_len * hidden_state_size), units=hidden_state_size)])

    def __call__(self, input):
        return self.encode(input)

    def encode(self, input):
        input = input.reshape(1, -1)
        embedding = self.model.predict(input).reshape(1, 1, -1)
        return self.lstm.predict(embedding)


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
        optimizer = optimizers.Adam()
                                    # lr=learning_rate  0.001 by default which is fine

        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy'  # TODO these are random, needs to be checked
                           # metrics=['accuracy']
                           )
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
                    Dense(6, input_shape=(hidden_state_size,), activity_regularizer=regularizers.l1(entropy_reg)),
                    Activation('softmax')
                ])
                model.compile(optimizer='adam',
                              loss='categorical_crossentropy'
                              # metrics=['accuracy']
                              )

                self.models.append(model)

    @property
    def dummy(self):
        return np.zeros(self.item_num)

    def forward(self, hidden_state, **kwargs):
        if not self.is_on:
            return self.dummy
        proposal = []
        for i in range(self.item_num):
            distribution = self.models[i].predict(hidden_state)[0]
            single_proposal = np.random.choice(np.arange(6), p=distribution)
            proposal.append(single_proposal)
        proposal = np.array(proposal)
        return proposal

    def train(self, x, y, sample_weight):
        if self.is_on:
            for i in range(self.item_num):
                y = convert_to_sparse(y[:, i], 6)
                # print('whats there for training in proposal policy??? \n', x.shape, y.shape, sample_weight.shape)
                self.models[i].train_on_batch(x, y, sample_weight=sample_weight)

    def get_weights(self):
        out = [model.get_weights() for model in self.models]
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
            lstm1 = LSTM(vocab_size, stateful=True, name='utter_lstm',
                         activity_regularizer=regularizers.l1(entropy_reg),
                         activation='softmax')(inputs)
            model = Model(inputs=inputs, outputs=[lstm1])
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy'
                          # TODO might be cool to use the one below (requires different shape in training)
                          # loss='sparse_categorical_crossentropy',
                          # metrics=['categorical_accuracy']
                          # sample_weight_mode="temporal"
                          )
            self.model = model

    @property
    def dummy(self):
        return np.zeros(self.utterance_len)

    @property
    def dummy_symbol(self):
        return np.zeros((1, 1, 1))

    def forward(self, hidden_state):
        if not self.is_on:
            utterance = self.dummy
        else:
            # self.input_is_valid(hidden_state)
            if hidden_state is not None:  # if hidden state is passed then we set is as a new LSTM state
                self.model.layers[1].states[0] = hidden_state
            utterance = [self.vocab[self.model.predict(self.dummy_symbol).argmax()]]
            for i in range(self.utterance_len - 1):
                last_symbol = np.full((1, 1, 1), utterance[-1])
                arg_max = self.model.predict(last_symbol).argmax()
                utterance.append(self.vocab[arg_max])
            utterance = np.array(utterance)

        # self.output_is_valid(utterance, (6,))
        return utterance

    def train(self, x, y, sample_weight):
        # TODO: why it doesnt work on a batch?!?!!!
        # TODO: x can be skipped, right?
        if self.is_on:
            # X = []
            # Y = []
            # SW = []
            for xx, yy, ssww in zip(x, y, sample_weight):
                inputs = [self.dummy_symbol] + yy[-1]
                ssww = np.array([ssww])
                yy_categorical = to_categorical(yy, num_classes=self.vocab_size).reshape(-1, 1, self.vocab_size)
                # self.model.train_on_batch(xx, yy)

                for xxx, yyy in zip(inputs, yy_categorical):
                    # X.append(xxx)
                    # Y.append(yyy)
                    # SW.append(ssww)
                    # X.append(xxx)
                    # Y.append(yyy)
                    # SW.append(ssww)
                    # print('shapesss', xxx.shape, np.array(yyy).shape, ssww.shape)
                    self.model.train_on_batch(xxx, np.array(yyy), sample_weight=ssww)
            # X = np.array(X)
            # Y = np.array(Y)
            # SW = np.array(SW)
            #
            # X = X.reshape(-1, 1, 1)
            # Y = Y.reshape(-1, 11)
            # SW = SW.reshape(-1)

            # X = np.array(X).reshape(-1, 1, 1)
            # Y = np.array(Y).reshape(-1, 11)
            # SW = np.array(SW).reshape(-1)
            # print('what are the shapes? X {}, Y {}, SW {}'.format(X.shape, Y.shape, SW.shape))
            # print('x', X[1])
            # print('y', Y[1])
            # print('sw', SW[1])
            # self.model.train_on_batch(X, Y, sample_weight=SW)
            # self.model.train_on_batch(X, Y, sample_weight=SW)
