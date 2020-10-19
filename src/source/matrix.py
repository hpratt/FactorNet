#!/usr/bin/env python

import ujson
import numpy

class RandomAccessMatrixFile:

    def __init__(self, path, nl = False, width = None):
        self.file = open(path, 'r')
        self.width = width
        firstline = self.file.readline()
        self.lengths = ujson.loads(firstline)
        if nl: self.lengths = [ x + 1 for x in self.lengths ]
        self.offsets = [ len(firstline) ]
        for x in self.lengths:
            self.offsets.append(self.offsets[-1] + x)
    
    def read(self, index, transform = lambda x: x):
        self.file.seek(self.offsets[index])
        r = numpy.array(transform(ujson.loads(self.file.read(self.lengths[index]))))
        if self.width is None: return r
        c = int(len(r) / 2)
        return r[c - self.width : c + self.width + 1]
    
    def close(self):
        self.file.close()
