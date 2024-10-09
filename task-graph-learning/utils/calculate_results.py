# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

# This script calculates the mean of the results obtained from the experiments executed with a single seed.

import click
import json
import glob

@click.command()
@click.option('--results', '-r', help='Path to the results file')
def main(results:str):
    json_files = glob.glob(results + '/*.json')
    metrics = {
        "Precision": 0,
        "Recall": 0,
        "F1": 0
    }
    for file in json_files:
        with open(file) as f:
            data = json.load(f)
            metrics["Precision"] += data["Precision"]
            metrics["Recall"] += data["Recall"]
            metrics["F1"] += data["F1"]
    metrics["Precision"] /= len(json_files)
    metrics["Recall"] /= len(json_files)
    metrics["F1"] /= len(json_files)

    print(metrics)

if __name__ == '__main__':
    main()