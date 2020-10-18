#!/usr/bin/env python

import numpy
import math
import ujson

from ..model.feature import feature_matrix

def generator(sequenceMatrixReader, featureMatrixReaders, zscoreJson, rDHSs, chromosomes, batch_size):
    allIndexes = [ x for x in rDHSs.indexesForChromosomes(chromosomes) ]
    with open(zscoreJson, 'r') as f:
        scores = ujson.load(f)
    pointer = 0
    while True:
        indexes = allIndexes[pointer : pointer + batch_size]
        forward_features = feature_matrix(
            [ numpy.array([ featureMatrixReader.read(i) for i in indexes ]) for featureMatrixReader in featureMatrixReaders ],
            numpy.array([ sequenceMatrixReader.read(i) for i in indexes ])
        )
        reverse_features = feature_matrix(
            [ numpy.array([ featureMatrixReader.read(i, lambda x: x[::-1]) for i in indexes ]) for featureMatrixReader in featureMatrixReaders ],
            numpy.array([ sequenceMatrixReader.read(i, lambda x: x[::-1]) for i in indexes ])
        )
        outputs = [ numpy.array([ (math.tanh(scores[i] / 2) + 1) / 2 ]) for i in indexes ]
        pointer += batch_size
        if pointer + batch_size >= len(allIndexes):
            pointer = 0
            allIndexes = [ x for x in rDHSs.indexesForChromosomes(chromosomes) ]
        yield ( [ forward_features, reverse_features ], outputs )
