"""
Resnet keras code.

Use resnet to implement face recognition algo.
"""
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, Input, MaxPooling2D


def identity_block(input_tensor, filters):
    """The identity block is the block that has no conv layer at shortcut."""
    x = Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, filters):
    """A block that has a conv layer at shortcut."""
    filter1, filter2 = filters

    x = Conv2D(filter1, 3, padding='same')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, 3)(x)
    x = BatchNormalization(axis=3)(x)

    shortcut = Conv2D(filter2, 3)(input_tensor)
    shortcut = BatchNormalization(axis=3)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet(input_shape: tuple=(128, 128, 3), classes: int=10):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape)(inputs)
    x = identity_block(x, 32)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = conv_block(x, [32, 64])
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = conv_block(x, [64, 128])
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu', name='dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', name='softmax')(x)
    model = Model(inputs, x)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


