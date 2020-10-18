#!/usr/bin/env python

import ujson
import numpy

class RandomAccessMatrixFile:

    def __init__(self, path):
        self.path = path
    
    def __enter__(self):
        self.file = open(self.path, 'r')
        firstline = self.file.readline()
        self.lengths = ujson.loads(firstline)
        self.offsets = [ len(firstline) ]
        for x in self.lengths:
            self.offsets.append(self.offsets[-1] + x)
        return self
    
    def read(self, index, transform = lambda x: x):
        self.file.seek(self.offsets[index])
        return numpy.array(transform(ujson.loads(self.file.readline())))
    
    def __exit__(self, *args):
        self.file.close()
