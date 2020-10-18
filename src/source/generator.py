#!/usr/bin/env python

import numpy
import ujson

from ..model.feature import feature_matrix
from .matrix import RandomAccessMatrixFile

def generator(sequenceMatrixFile, featureMatrixFiles, zscoreJson, rDHSs, chromosomes, batch_size):
    sequenceMatrixReader = RandomAccessMatrixFile(sequenceMatrixFile)
    featureMatrixReaders = RandomAccessMatrixFile(featureMatrixFiles)
    allIndexes = rDHSs.indexesForChromosomes(chromosomes)
    with open(zscoreJson, 'r') as f:
        scores = ujson.load(f)
    pointer = 0
    while True:
        indexes = allIndexes[pointer : pointer + batch_size]
        forward_features = feature_matrix(
            [ numpy.array([ featureMatrixFile.read(i) for i in indexes ]) for featureMatrixFile in featureMatrixFiles ],
            numpy.array([ sequenceMatrixReader.read(i) for i in indexes ])
        )
        reverse_features = feature_matrix(
            [ numpy.array([ featureMatrixFile.read(i, lambda x: x[::-1]) for i in indexes ]) for featureMatrixFile in featureMatrixFiles ],
            numpy.array([ sequenceMatrixReader.read(i, lambda x: x[::-1]) for i in indexes ])
        )
        outputs = [ scores[i] for i in indexes ]
        pointer += batch_size
        if pointer >= len(allIndexes): pointer = 0
        yield ( [ forward_features, reverse_features ], outputs )
