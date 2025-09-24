#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sept 28 2025
@authors: Viale J / Cougouluegne T
"""

# Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist
from keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# load data
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()

trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

trainData = trainData.astype('float32')
testData = testData.astype('float32')

trainData = trainData / 255.0
testData = testData / 255.0

trainLabels = np_utils.to_categorical(trainLabels)
testLabels = np_utils.to_categorical(testLabels)
num_classes = testLabels.shape[1]
print(trainData.shape)

# LeNet model
model = Sequential()
model.add(Conv2D(20,(5,5), input_shape = (28,28,1), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(40,(5,5), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(400,activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())

model.fit(trainData, trainLabels, batch_size=128, epochs=20, verbose=1)

#save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('lenet_weights.weights.h5')    

scores = model.evaluate(testData,testLabels,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
