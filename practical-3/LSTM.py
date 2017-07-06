import numpy as np
import tensorflow as tf

numEpochs = 100
totalSeriesLength = 50000
truncatedBackpropLength = 15
stateSize = 4
numClasses = 2
echoStep = 3
batchSize = 5
numBatches = totalSeriesLength // batchSize // truncatedBackpropLength


def generateData():
    x = np.array(np.random.choice(2, totalSeriesLength, p=[0.5, 0.5]))
    y = np.roll(x, echoStep)
    y[0:echoStep] = 0

    x = x.reshape((batchSize, -1))
    y = y.reshape((batchSize, -1))

    return (x, y)


batchXPlaceholder = tf.placeholder(tf.float32, [batchSize, truncatedBackpropLength])
batchYPlaceholder = tf.placeholder(tf.int32, [batchSize, truncatedBackpropLength])

initState = tf.placeholder(tf.float32, [batchSize, stateSize])

W2 = tf.Variable(np.random.rand(stateSize, numClasses), dtype=tf.float32)
b2 = tf.Variable(np.random.rand(1, numClasses), dtype=tf.float32)

inputsSeries = tf.split(batchXPlaceholder, truncatedBackpropLength, 1)
labelsSeries = tf.unstack(batchYPlaceholder, axis=1)

# Forward pass
cell = tf.contrib.rnn.BasicRNNCell(stateSize)
statesSeries, currentState = tf.contrib.rnn.static_rnn(cell, inputsSeries, initState)

logitsSeries = [tf.matmul(state, W2) + b2 for state in statesSeries]
predictionsSeries = [tf.nn.softmax(logits) for logits in logitsSeries]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in
          zip(logitsSeries, labelsSeries)]
totalLoss = tf.reduce_mean(losses)

trainStep = tf.train.AdagradOptimizer(0.3).minimize(totalLoss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epochIdx in range(numEpochs):
        x, y = generateData()
        _currentState = np.zeros((batchSize, stateSize))

        print("New data, epoch", epochIdx)

        for batchIdx in range(numBatches):
            startIdx = batchIdx * truncatedBackpropLength
            endIdx = startIdx + truncatedBackpropLength

            batchX = x[:, startIdx:endIdx]
            batchY = y[:, startIdx:endIdx]

            _totalLoss, _trainStep, _currentState, _predictionsSeries = sess.run(
                [totalLoss, trainStep, currentState, predictionsSeries],
                feed_dict={batchXPlaceholder: batchX, batchYPlaceholder: batchY, initState: _currentState})

            if batchIdx % 100 == 0:
                print("Step",batchIdx, "Loss", _totalLoss)
