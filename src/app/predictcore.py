#!/usr/bin/env python

import sys
import os
import ujson

from ..model.deserialize import deserializeModel
from ..source.generator import generator
from ..source.matrix import RandomAccessMatrixFile
from ..utils.rDHS import RDHSSet
from .traincore import TRAINING_CHROMOSOME_DEFAULTS, VALIDATION_CHROMOSOME_DEFAULTS

def predict_core(rDHSs, featureJsons, sequenceJson, signalZScores, modelDirectory, outputFile, batchSize = 1000):

    print("loading model...", file = sys.stderr)
    model = deserializeModel(os.path.join(modelDirectory, "model.json"), os.path.join(modelDirectory, "best_model.hdf5"))

    print("loading regions...", file = sys.stderr)
    rDHSs = RDHSSet(rDHSs)
    sequenceMatrixReader = RandomAccessMatrixFile(sequenceJson)
    featureMatrixReaders = [ RandomAccessMatrixFile(featureJson) for featureJson in featureJsons ]

    print("making predictions...", file = sys.stderr)
    chromosomes = TRAINING_CHROMOSOME_DEFAULTS.union(VALIDATION_CHROMOSOME_DEFAULTS)
    dataGenerator = generator(sequenceMatrixReader, featureMatrixReaders, signalZScores, rDHSs, chromosomes, batchSize, True)
    results = model.predict(dataGenerator, batchSize, steps = len(rDHSs))
    batches = []
    for _ in range(len(rDHSs) / batchSize + 1):
        batches += next(dataGenerator)

    print("writing results", file = sys.stderr)
    with open(outputFile, 'w') as o:
        for i, region in enumerate(rDHSs.regions):
            o.write("%s\t%.3f\t%.3f\n" % ('\t'.join([ str(x) for x in region ]), results[i], batches[i]))
