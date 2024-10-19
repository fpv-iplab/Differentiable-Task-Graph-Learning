# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import click
import glob

@click.command()
@click.option('--results', '-r', help='Path to the results file')
def main(results:str):
    txt_files = glob.glob(results + '/*.txt')
    accuracy = 0
    for file in txt_files:
        with open(file) as f:
            accuracy += float(f.readline())
    accuracy /= len(txt_files)

    print(accuracy)

if __name__ == '__main__':
    main()