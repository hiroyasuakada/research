"""
#This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""

import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np
from matplotlib import pyplot as plt


def Convolutional_LSTM(n_frames, width, height, channels):
    # take input movies of shape ( )
    input = layers.Input(shape=(n_frames, width, height, channels))

    x = layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True)(input)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same', data_format='channels_last')(x)
    x = layers.Activation('sigmoid')(x)

    conv_lstm = tf.keras.models.Model(input=input, output=x)
    conv_lstm.compile(optimizer='adadelta', loss='binary_crossentropy')

    # return a movie of identical shape
    return conv_lstm


def


