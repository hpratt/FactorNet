#!/usr/bin/env python

from keras.models import model_from_json

def deserializeModel(json, weights):
    with open(json, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weights)
    return model
