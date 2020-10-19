#!/usr/bin/env python

import numpy
import math
import ujson
import random

from ..model.feature import feature_matrix

def generator(sequenceMatrixReader, featureMatrixReaders, zscoreJson, rDHSs, chromosomes, batch_size, sortedI = False):
    indexes = [ x for x in rDHSs.indexesForChromosomes(chromosomes) ]
    random.shuffle(indexes)
    if sortedI: indexes = sorted(indexes)
    with open(zscoreJson, 'r') as f:
        scores = ujson.load(f)
    pointer = 0
    while True:
        forward_features = []; reverse_features = []; outputs = []
        while len(forward_features) < batch_size:
            try:
                ff = [ featureMatrixReader.read(indexes[pointer]) for featureMatrixReader in featureMatrixReaders ] + [ sequenceMatrixReader.read(indexes[pointer]) ]
                rr = [ featureMatrixReader.read(indexes[pointer], lambda x: x[::-1]) for featureMatrixReader in featureMatrixReaders ] + [ sequenceMatrixReader.read(indexes[pointer], lambda x: x[::-1]) ]
                oo = numpy.array([ 1 if math.tanh(scores[indexes[pointer]] / 2) > 0.8 else 0 ])
                forward_features.append(ff); reverse_features.append(rr); outputs.append(oo)
            except:
                pass
            pointer += 1
            if pointer + batch_size >= len(indexes):
                pointer = 0
                indexes = [ x for x in rDHSs.indexesForChromosomes(chromosomes) ]
                random.shuffle(indexes)
                if sortedI: indexes = sorted(indexes)
        forward_features = feature_matrix(
            [ numpy.array([ forward_features[j][i] for j in range(len(forward_features)) ]) for i in range(len(forward_features[0]) - 1) ],
            numpy.array([ forward_features[i][-1] for i in range(len(forward_features)) ])
        )
        reverse_features = feature_matrix(
            [ numpy.array([ reverse_features[j][i] for j in range(len(reverse_features)) ]) for i in range(len(reverse_features[0]) - 1) ],
            numpy.array([ reverse_features[i][-1] for i in range(len(reverse_features)) ])
        )
        pointer += batch_size
        yield ( [ forward_features, reverse_features ], outputs )
