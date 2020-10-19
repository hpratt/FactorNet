#!/usr/bin/env python

import sys
import argparse

from .traincore import train_core
from .predictcore import predict_core

def run_train_core(args):
    train_core(
        args.rDHSs,
        args.feature_files if args.feature_files is not None else [],
        args.sequence_matrix,
        args.signal_z_scores,
        args.output_directory
    )

def run_predict_core(args):
    predict_core(
        args.rDHSs,
        args.feature_files,
        args.sequence_matrix, 
        args.signal_z_scores,
        args.model_directory,
        args.output_file,
        limit = args.limit
    )

def main():
    parser = argparse.ArgumentParser(description = "Trains a TF binding model.")
    subparsers = parser.add_subparsers()

    core = subparsers.add_parser("traincore", help = "core model, without extended metadata or cell-type-level features")
    core.add_argument("--rDHSs", type = str, help = "path to rDHS regions to which features correspond", required = True)
    core.add_argument("--feature-files", type = str, nargs = '+', help = "path to feature matrices in JSON format")
    core.add_argument("--sequence-matrix", type = str, help = "path to one-hot sequence matrix for rDHSs", required = True)
    core.add_argument("--signal-z-scores", type = str, help = "path to BED file containing TF Z-scores for each rDHS", required = True)
    core.add_argument("--output-directory", type = str, help = "path to output directory")
    core.set_defaults(func = run_train_core)

    predict = subparsers.add_parser("predictcore", help = "use core model, without extended metadata or cell-type-level features, for prediction")
    predict.add_argument("--rDHSs", type = str, help = "path to rDHS regions to which features correspond", required = True)
    predict.add_argument("--feature-files", type = str, nargs = '+', help = "path to feature matrices in JSON format")
    predict.add_argument("--sequence-matrix", type = str, help = "path to one-hot sequence matrix for rDHSs", required = True)
    predict.add_argument("--signal-z-scores", type = str, help = "path to BED file containing TF Z-scores for each rDHS", required = True)
    predict.add_argument("--model-directory", type = str, help = "path to model directory as produced by traincore", required = True)
    predict.add_argument("--output-file", type = str, help = "path to output file", required = True)
    predict.add_argument("--limit", type = int, help = "if set, limits predictions to the first n rDHSs", default = None)
    predict.set_defaults(func = run_predict_core)

    args = parser.parse_args()
    args.func(args)

    return 0

if __name__ == "__main__":
    sys.exit(main())
