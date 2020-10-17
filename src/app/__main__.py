#!/usr/bin/env python

import sys
import argparse

from .traincore import train_core

def run_train_core(args):
    train_core(
        args.rDHSs,
        args.feature_files if args.feature_files is not None else [],
        args.sequence_matrix,
        args.signal_z_scores,
        args.output_directory
    )

def main():
    parser = argparse.ArgumentParser(description = "Trains a TF binding model.")
    subparsers = parser.add_subparsers()

    core = subparsers.add_parser("core", help = "core model, without extended metadata or cell-type-level features")
    core.add_argument("--rDHSs", type = str, help = "path to rDHS regions to which features correspond", required = True)
    core.add_argument("--feature-files", type = str, nargs = '+', help = "path to feature matrices in JSON format")
    core.add_argument("--sequence-matrix", type = str, help = "path to one-hot sequence matrix for rDHSs", required = True)
    core.add_argument("--signal-z-scores", type = str, help = "path to BED file containing TF Z-scores for each rDHS", required = True)
    core.add_argument("--output-directory", type = str, help = "path to output directory")
    core.set_defaults(func = run_train_core)

    args = parser.parse_args()
    args.func(args)

    return 0

if __name__ == "__main__":
    sys.exit(main())
