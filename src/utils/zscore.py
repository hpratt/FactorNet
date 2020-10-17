#!/usr/bin/env python2

import numpy
import math

class ZScores:

    def __init__(self, path, rDHSs):
        self.path = path
        self.rDHSs = rDHSs

    def valuesForChromosome(self, chromosomes, transform = lambda x: x):
        accessions = set()
        for chromosome in chromosomes:
            accessions = accessions.union(self.rDHSs.accessionsForChromosome(chromosome))
        with open(self.path, 'r') as f:
            values = { x.strip().split()[-2]: transform(float(x.strip().split()[-1])) for x in f }
        indexes = sorted(accessions, key = lambda x: self.rDHSs.accessionIndexMap[x])
        return numpy.array([ values[x] for x in indexes ])

    def tanhValuesForChromosome(self, chromosomes):
        return self.valuesForChromosome(chromosomes, math.tanh)
