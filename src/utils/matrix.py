#!/usr/bin/env python2

import numpy
import json

class FeatureMatrix:

    def __init__(self, path, rDHSs):
        self.path = path
        self.rDHSs = rDHSs
    
    def extractFeaturesFromChromosomes(self, chromosomes, transform = lambda x: x):
        indexes = set()
        for chromosome in chromosomes:
            indexes = indexes.union(self.rDHSs.indexesForChromosome(chromosome))
        indexes = sorted(indexes)
        with open(self.path, 'r') as f:
            j = json.load(f)
        return numpy.array([ transform(j[i]) for i in indexes ])

    def extractReverseFeaturesFromChromosomes(self, chromosomes):
        return self.extractFeaturesFromChromosomes(chromosomes, lambda x: x[::-1])
