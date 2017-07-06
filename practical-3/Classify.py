from DataPrep import *
import numpy as np
import os
import _pickle as pickle
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU
from keras.preprocessing.sequence import pad_sequences

print('Collecting data...')

if not os.path.isfile('pickledData.p'):
    data = initData()
    labelsText = getLabelsAndText(data)
    pickle.dump(labelsText, open('pickledData.p', 'wb'))
else:
    labelsText = pickle.load(open('pickledData.p', 'rb'))

trainData = labelsText[:1789]
testData = labelsText[1789:]

trainX = [train[1] for train in trainData]
trainX = pad_sequences(trainX, maxlen=512)
trainY = [train[0] for train in trainData]
trainY = np.array(trainY)

print('Size of training data (input): ' + str(trainX.shape))
print('Size of training data (output): ' + str(trainY.shape))

testX = [test[1] for test in testData]
testX = pad_sequences(testX, maxlen=512)
testY = [test[0] for test in testData]
testY = np.array(testY)

print('Size of training data (input): ' + str(testX.shape))
print('Size of training data (output): ' + str(testY.shape))

print('Building model...')

model = Sequential()
model.add(Embedding(512, 256, input_length=512))
model.add(GRU(64, activation='tanh'))
model.add(Dense(8, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('Training model...')

model.fit(trainX, trainY, verbose=2, epochs=70, batch_size=128)

print('Evaluating model...')

score = model.evaluate(testX, testY)

print('Score was: ' + str(score))