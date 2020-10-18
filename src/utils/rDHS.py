#!/usr/bin/env python2

class RDHSSet:

    def __init__(self, path):
        with open(path, 'r') as f:
            self.regions = [ tuple(line.strip().split()[:4]) for line in f ]
        self.indexmap = { x: i for i, x in enumerate(self.regions) }
        self.accessionIndexMap = { x[-1]: i for i, x in enumerate(self.regions) }

    def indexesForChromosome(self, chromosome):
        return { self.indexmap[x] for x in self.regions if x[0] == chromosome }
    
    def indexesForChromosomes(self, chromosomes):
        r = set()
        for x in chromosomes:
            r = r.union(self.indexesForChromosome(x))
        return r

    def accessionsForChromosome(self, chromosome):
        return { x[-1] for x in self.regions if x[0] == chromosome }
