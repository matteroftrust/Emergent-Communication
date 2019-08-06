import numpy as np

from .utils import convert_to_sparse

from tensorflow.python.keras import Input, regularizers, optimizers
from tensorflow.python.keras.layers import Dense, Activation, LSTM, concatenate, Lambda
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.models import Model
from keras import backend as K
import tensorflow as tf

# from keras.utils import plot_model plot_model(model, to_file='model.png')  !!!!!!!!

"""
TODO:
- regularization: should be l1 or l2?
- fix losses
- utterance policy to be fixed maybe with RepeatVector https://stackoverflow.com/questions/51749404/how-to-connect-lstm-layers-in-keras-repeatvector-or-return-sequence-true
"""

trainable = True

class AllInOneModel:
    def __init__(self, hidden_state_size, context_len, utterance_len, proposal_len, term_entropy=.05, prop_entropy=.05, utter_entropy=.001):
        context_input = Input(batch_shape=(1, context_len), name='context_input')
        utterance_input = Input(batch_shape=(1, utterance_len), name='utterance_input')
        proposal_input = Input(batch_shape=(1, proposal_len), name='proposal_input')

        context_embedding = Embedding(input_dim=11, output_dim=hidden_state_size, trainable=False, name='context_embedd')(context_input)
        utterance_embedding = Embedding(input_dim=11, output_dim=hidden_state_size, trainable=False, name='utterance_embedd')(utterance_input)
        proposal_embedding = Embedding(input_dim=11, output_dim=hidden_state_size, trainable=False, name='proposal_embedd')(proposal_input)

        context_lstm = LSTM(units=hidden_state_size, trainable=trainable, name='context_lstm')(context_embedding)
        utterance_lstm = LSTM(units=hidden_state_size, trainable=trainable, name='utterance_lstm')(utterance_embedding)
        proposal_lstm = LSTM(units=hidden_state_size, trainable=trainable, name='proposal_lstm')(proposal_embedding)

        merged = concatenate([context_lstm, utterance_lstm, proposal_lstm])
        hidden_state = Dense(hidden_state_size, activation='relu', trainable=trainable, name='hidden_state')(merged)

        termination_policy = Dense(1, activation='sigmoid', name='termination_policy',
                                   activity_regularizer=regularizers.l1(term_entropy))(hidden_state)

        dummy_input = Lambda(self.dummy_symbol_input)(hidden_state)
        utterance_policy = Dense(66, name='utterance_policy', activation='softmax',
                                 activity_regularizer=regularizers.l1(utter_entropy), trainable=False,)(dummy_input)

        proposal_policy_0 = Dense(6, activation='softmax', name='proposal_policy_0',
                                  activity_regularizer=regularizers.l1(prop_entropy))(hidden_state)
        proposal_policy_1 = Dense(6, activation='softmax', name='proposal_policy_1',
                                  activity_regularizer=regularizers.l1(prop_entropy))(hidden_state)
        proposal_policy_2 = Dense(6, activation='softmax', name='proposal_policy_2',
                                  activity_regularizer=regularizers.l1(prop_entropy))(hidden_state)

        model = Model(inputs=[context_input, utterance_input, proposal_input],
                      outputs=[termination_policy, utterance_policy, proposal_policy_0, proposal_policy_1, proposal_policy_2]
                      )
        losses = {
            "termination_policy": "binary_crossentropy",
            "utterance_policy": "mean_squared_error",
            "proposal_policy_0": "categorical_crossentropy",
            "proposal_policy_1": "categorical_crossentropy",
            "proposal_policy_2": "categorical_crossentropy"
        }

        model.compile(optimizer='adam', loss=losses)
        self.model = model

    @property
    def dummy(self):
        return np.zeros(self.utterance_len)

    @property
    def dummy_symbol(self):
        return np.zeros((1, 1, 1))

    def forward(self, context, utterance, proposal):
        self.model.predict([context, utterance, proposal])

    def dummy_symbol_input(self, *args, **kwargs):
        c = K.constant(self.dummy_symbol)
        return c

    def predict(self, context, utterance, proposal, test=False):
        # print('what came?', context, utterance, proposal)
        termination_policy, utterance_policy, proposal_policy_0, proposal_policy_1, proposal_policy_2 = self.model.predict([context, utterance, proposal])
        y = [termination_policy, utterance_policy, proposal_policy_0, proposal_policy_1, proposal_policy_2]

        # print('what was predicted', termination_policy, utterance_policy, proposal_policy_0, proposal_policy_1, proposal_policy_2)
        termination_policy = float(termination_policy)
        if test:
            termination = [True, False][termination_policy < 0.5]
        else:
            termination = np.random.choice([True, False], p=[termination_policy, 1 - termination_policy])
        utterance = np.zeros(6)
        proposal = []
        for distribution in [proposal_policy_0, proposal_policy_1, proposal_policy_2]:
            if test:
                single_proposal = distribution.argmax()
            else:
                single_proposal = np.random.choice(np.arange(6), p=distribution.reshape(-1))
            proposal.append(single_proposal)
        sparse_proposal = convert_to_sparse(np.array(proposal), 6)

        y = [np.array([termination], dtype=int), utterance_policy, sparse_proposal[0], sparse_proposal[1], sparse_proposal[2]]
        for i in range(len(y)):
            y[i] = y[i].reshape(1, -1)

        return termination, utterance, proposal, y

    def train(self, x, y, sw):

        self.model.fit(x, y, sample_weight=[sw] * 5, verbose=1)
