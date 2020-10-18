#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import pickle

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import keras.metrics

from ..utils.rDHS import RDHSSet
from ..utils.zscore import ZScores
from ..model.feature import forward_features, reverse_features
from ..model.core import core_model
from ..source.generator import generator

TRAINING_CHROMOSOME_DEFAULTS = { "chr1", "chr2", "chr4", "chr6", "chr7", "chr8", "chr9", "chr10", "chr13", "chr15", "chr17", "chr18", "chr20", "chr21", "chr22" }
VALIDATION_CHROMOSOME_DEFAULTS = { "chr3", "chr5", "chr11", "chr12", "chr14", "chr16", "chr19" }
DEFAULT_FILTER_COUNT = 48
DEFAULT_RECURRENT_LAYER_COUNT = 1
DEFAULT_DENSE_LAYER_COUNT = 1
DEFAULT_DROPOUT_RATE = 0.5
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCH_LIMIT = 100
DEFAULT_PATIENCE = 20
DEFAULT_BATCH_SIZE = 50000

def train_core(
    rDHSs,
    featureJsons,
    sequenceJson,
    signalZScores,
    outputDir,
    trainingChromosomes = TRAINING_CHROMOSOME_DEFAULTS,    
    validationChromosomes = VALIDATION_CHROMOSOME_DEFAULTS,
    filterCount = DEFAULT_FILTER_COUNT,
    recurrentLayerCount = DEFAULT_RECURRENT_LAYER_COUNT,
    denseLayerCount = DEFAULT_DENSE_LAYER_COUNT,
    dropoutRate = DEFAULT_DROPOUT_RATE,
    learningRate = DEFAULT_LEARNING_RATE,
    epochLimit = DEFAULT_EPOCH_LIMIT,
    patience = DEFAULT_PATIENCE,
    batchSize = DEFAULT_BATCH_SIZE
):

    print("loading regions...", file = sys.stderr)
    rDHSs = RDHSSet(rDHSs)

    print("compiling model...", file = sys.stderr)
    model = core_model(1, len(forward_train[0]), len(featureJsons), filterCount, recurrentLayerCount, denseLayerCount, dropoutRate)
    model.compile(optimizer = Adam(lr = learningRate), loss = 'mean_squared_error', metrics = [ 'mean_squared_error' ])
    model.summary()

    print("running at most {epochs} epochs".format(epochs = epochLimit), file = sys.stderr)
    checkpointer = ModelCheckpoint(
        filepath = os.path.join(outputDir, 'best_model.hdf5'),
        verbose = 1, save_best_only = True
    )
    earlystopper = EarlyStopping(monitor = 'val_loss', patience = patience, verbose = 1)
    train_samples_per_epoch = len(forward_train) / epochLimit
    history = model.fit_generator(
        generator(sequenceJson, featureJsons, signalZScores, rDHSs, trainingChromosomes, batchSize),
        training_outputs,
        epochs = epochLimit,
        validation_data = generator(sequenceJson, featureJsons, signalZScores, rDHSs, validationChromosomes, batchSize),
        validation_steps = len(forward_valid),
        callbacks = [checkpointer, earlystopper]
    )

    print("saving model...", file = sys.stderr)
    model.save_weights(os.path.join(outputDir, 'final_model.hdf5'), overwrite = True)
    with open(os.path.join(outputDir, '/history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
