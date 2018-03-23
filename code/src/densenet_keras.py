"""
Densenet keras code.

Use dense to implement face recognition algo.
"""
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, Input, MaxPooling2D


def densenet(input_shape: tuple, classes: int):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, padding='same', input_shape=input_shape)(inputs)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')
    x1 = Conv2D(32, 3, padding='same')(input)

    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model