from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence

import numpy as np

# Training = 500k episodes
# 1 episode = 128 games

# Test = 5 episodes

# hyperparameters

itempool = [2, 7, 3]
prop_utilities = [6, 2, 1]
utterance_prev = [2, 4, 2, 1, 3, 5]
item_context = np.array([6, 2, 1, 2, 4, 2, 1, 3, 5])
VOCAB_SIZE = 11
DIM_SIZE = 100
MAX_LENGTH = 6  # maximal allowed length of produced utterances
lambda_termination = 0.05  # entropy reguralization weight hyperparameter for termination policy
lambda_proposal = 0.05  # entropy reguralization weight hyperparameter for proposal policy
lambda_utterance = 0.001  # entropy reguralization weight hyperparameter for linguistic utterance policy
SMOOTHING_CONST = 0.7

model = Sequential()
model.add(Embedding(VOCAB_SIZE, DIM_SIZE))
model.add(LSTM(100))
model.compile(optimizer='adam', loss='mse')
print(model.summary())
utterance_prev = np.reshape(utterance_prev, (1, 6))
output_array = model.predict(item_context)


# Policy Terminate

model = Sequential()
model.add(Dense(2, input_dim=100, activation="sigmoid"))
model.compile(optimizer='adam', loss='mse')

print(model.summary())
terminate = model.predict(output_array)
print(terminate)

# model.fit(state, reward_value, epochs=1, verbose=0)
