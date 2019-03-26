from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

#Training = 500k episodes
#1 episode = 128 games

#Test = 5 episodes 

itempool = [2,7,3]
prop_utilities = [6,2,1]
utterance_prev = [2,4,2,1,3,5]
item_context = [[6,2,1],[2,4,2,1,3,5]]

vocab_size = 11
dim_size = 100
max_length = 6

model = Sequential()
model.add(Embedding(vocab_size,dim_size,input_length = max_length))
model.add(LSTM(100))
model.compile(optimizer='adam', loss='mse')
print(model.summary())

model.fit(state, reward_value, epochs=1, verbose=0)