#!/usr/bin/env python2

import numpy

from .matrix import FeatureMatrix

def reverseComplement(sequence):
    return numpy.array([ x[::-1] for x in sequence ][::-1])

class SequenceMatrix(FeatureMatrix):

    def __init__(self, path, rDHSs):
        FeatureMatrix.__init__(self, path, rDHSs)
    
    def extractReverseComplementFromChromosomes(self, chromosomes):
        return self.extractFeaturesFromChromosomes(chromosomes, reverseComplement)
