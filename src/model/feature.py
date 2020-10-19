#!/usr/bin/env python2

import numpy

from ..utils.sequence import SequenceMatrix
from ..utils.matrix import FeatureMatrix

def forward_features(featureJsons, sequenceJson, rDHSs, chromosomes):
    features = SequenceMatrix(sequenceJson, rDHSs).extractFeaturesFromChromosomes(chromosomes)
    for featureJson in featureJsons:
        features[:,:,-1] = FeatureMatrix(featureJson, rDHSs).extractFeaturesFromChromosomes(chromosomes)
    return features

def reverse_features(featureJsons, sequenceJson, rDHSs, chromosomes):
    features = SequenceMatrix(sequenceJson, rDHSs).extractReverseComplementFromChromosomes(chromosomes)
    for featureJson in featureJsons:
        features[:,:,-1] = FeatureMatrix(featureJson, rDHSs).extractReverseFeaturesFromChromosomes(chromosomes)
    return features

def feature_matrix(featureMatrices, sequenceMatrix):
    for featureMatrix in featureMatrices:
        sequenceMatrix = numpy.array([
            numpy.array([ numpy.append(sequenceMatrix[k][j], [ featureMatrix[k][i][j] for i in range(len(featureMatrix[k])) ]) for j in range(len(featureMatrix[k][0])) ])
            for k in range(len(featureMatrix))
        ])
    return sequenceMatrix
