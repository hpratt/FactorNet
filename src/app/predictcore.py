#!/usr/bin/env python

import sys
import os
import ujson

from ..model.deserialize import deserializeModel
from ..source.generator import generator
from ..source.matrix import RandomAccessMatrixFile
from ..utils.rDHS import RDHSSet
from .traincore import TRAINING_CHROMOSOME_DEFAULTS, VALIDATION_CHROMOSOME_DEFAULTS

def predict_core(rDHSs, featureJsons, sequenceJson, signalZScores, modelDirectory, outputFile, batchSize = 1000, limit = None):

    print("loading model...", file = sys.stderr)
    model = deserializeModel(os.path.join(modelDirectory, "model.json"), os.path.join(modelDirectory, "best_model.hdf5"))

    print("loading regions...", file = sys.stderr)
    rDHSs = RDHSSet(rDHSs)
    sequenceMatrixReader = RandomAccessMatrixFile(sequenceJson, width = 150)
    featureMatrixReaders = [ RandomAccessMatrixFile(featureJson) for featureJson in featureJsons ] if featureJsons is not None else []

    print("making predictions...", file = sys.stderr)
    chromosomes = TRAINING_CHROMOSOME_DEFAULTS.union(VALIDATION_CHROMOSOME_DEFAULTS)
    dataGenerator = generator(sequenceMatrixReader, featureMatrixReaders, signalZScores, rDHSs, chromosomes, batchSize, True)
    results = model.predict(dataGenerator, batchSize, verbose = 1, steps = (len(rDHSs) if limit is None else limit) / batchSize)

    print("loading original values...", file = sys.stderr)
    dataGenerator = generator(sequenceMatrixReader, featureMatrixReaders, signalZScores, rDHSs, chromosomes, batchSize, True)
    batches = []; limit = int((len(rDHSs) if limit is None else limit) / batchSize + 1)
    for i in range(limit):
        print("   ... for batch %d of %d ..." % (i + 1, limit))
        batches += next(dataGenerator)[1]

    print("writing results", file = sys.stderr)
    with open(outputFile, 'w') as o:
        for i, result in enumerate(results):
            o.write("%s\t%.3f\t%.3f\n" % ('\t'.join([ str(x) for x in rDHSs.regions[i] ]), result, batches[i][0]))
