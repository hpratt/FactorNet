#!/usr/bin/env python2

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Layer, Maximum, Input
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed

from .util import get_output

def core_model(num_tfs, width, feature_count, filter_count, recurrent_layer_count, dense_layer_count, dropout_rate, filter_length = 48):
    forward_input = Input(shape=(width, 4 + feature_count,))
    reverse_input = Input(shape=(width, 4 + feature_count,))
    if recurrent_layer_count < 0:
        hidden_layers = [
            Convolution1D(
                input_dim = 4 + feature_count,
                nb_filter = filter_count,
                filter_length = filter_length,
                border_mode = 'valid',
                activation = 'relu',
                subsample_length = 1
            ),
            Dropout(0.1),
            TimeDistributed(Dense(filter_count, activation = 'relu')),
            GlobalMaxPooling1D(),
            Dropout(dropout_rate),
            Dense(dense_layer_count, activation = 'relu'),
            Dropout(dropout_rate),
            Dense(num_tfs, activation = 'sigmoid')
        ]
    elif recurrent_layer_count == 0:
        hidden_layers = [
            Convolution1D(
                input_dim = 4 + feature_count,
                nb_filter = filter_count,
                filter_length = filter_length,
                border_mode = 'valid',
                activation = 'relu',
                subsample_length = 1
            ),
            Dropout(0.1),
            TimeDistributed(Dense(filter_count, activation = 'relu')),
            MaxPooling1D(pool_length = filter_length / 2, stride = filter_length / 2),
            Dropout(dropout_rate),
            Flatten(),
            Dense(dense_layer_count, activation = 'relu'),
            Dropout(dropout_rate),
            Dense(num_tfs, activation = 'sigmoid')
        ]
    else:
        hidden_layers = [
            Convolution1D(
                input_dim = 4 + feature_count,
                nb_filter = filter_count,
                filter_length = filter_length,
                border_mode = 'valid',
                activation = 'relu',
                subsample_length = 1
            ),
            Dropout(0.1),
            TimeDistributed(Dense(filter_count, activation = 'relu')),
            MaxPooling1D(pool_length = filter_length / 2, stride = filter_length / 2),
            Bidirectional(LSTM(recurrent_layer_count, dropout_W = 0.1, dropout_U = 0.1, return_sequences = True)),
            Dropout(dropout_rate),
            Flatten(),
            Dense(dense_layer_count, activation = 'relu'),
            Dropout(dropout_rate),
            Dense(num_tfs, activation = 'sigmoid')
        ]
    forward_output = get_output(forward_input, hidden_layers)     
    reverse_output = get_output(reverse_input, hidden_layers)
    output = Maximum()([forward_output, reverse_output ])
    return Model(input = [ forward_input, reverse_input ], output = output)
