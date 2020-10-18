#!/usr/bin/env python

import ujson
import numpy

class RandomAccessMatrixFile:

    def __init__(self, path):
        self.file = open(path, 'r')
        firstline = self.file.readline()
        self.lengths = ujson.loads(firstline)
        self.offsets = [ len(firstline) ]
        for x in self.lengths:
            self.offsets.append(self.offsets[-1] + x)
    
    def read(self, index, transform = lambda x: x):
        self.file.seek(self.offsets[index])
        return numpy.array(transform(ujson.loads(self.file.read(self.lengths[index]))))
    
    def close(self):
        self.file.close()
