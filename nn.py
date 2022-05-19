from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import Input
import numpy as np

from keras.optimizer_v2.adam import Adam


def get_model_confidence_level(X, y):
    """
    input needs to be a np.array
    """

    cards_num = len(X[0])

    # define the keras model
    model = Sequential()
    model.add(Input(shape=(cards_num,)))
    model.add(Dense(cards_num * 4, activation='relu'))
    model.add(Dense(cards_num * 2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # print("what the model is expecting")
    # [print("inputs i.shape, i.dtype", i.shape, i.dtype) for i in model.inputs]
    # [print("outputs o.shape, o.dtype", o.shape, o.dtype) for o in model.outputs]
    # [print("layers l.name, l.input_shape, l.dtype", l.name, l.input_shape, l.dtype) for l in model.layers]

    # compile the keras model
    opt = Adam(learning_rate=0.0008)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(X, y, epochs=30, batch_size=100)

    return model


def get_model_agent(X, y):
    """
    input needs to be a np.array
    """

    input_dim = len(X[0])
    print("cards_num:", input_dim)

    # define the keras model
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(input_dim * 2, activation='relu'))
    model.add(Dense(input_dim, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    # print("what the model is expecting")
    # [print("inputs i.shape, i.dtype", i.shape, i.dtype) for i in model.inputs]
    # [print("outputs o.shape, o.dtype", o.shape, o.dtype) for o in model.outputs]
    # [print("layers l.name, l.input_shape, l.dtype", l.name, l.input_shape, l.dtype) for l in model.layers]

    # compile the keras model
    opt = Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(X, y, epochs=40, batch_size=2)

    return model
