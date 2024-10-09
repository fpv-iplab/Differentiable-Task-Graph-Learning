# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import json
import click
import os

try:
    from taskgraph.task_graph_learning import baseline_transition_graph, save_graph_as_svg
except:
    raise Exception("You need to install the TGML library. Please read the README.md file.")

@click.command()
@click.option('--keysteps', help='File JSON containing the keysteps', required=True)
@click.option('--baseline_folder_output', '-bfo', help='Baseline folder output', required=True)
@click.option('--max_length', '-ml', help='Max length of the sequences', default=-1)
def main(keysteps:str, baseline_folder_output:str, max_length:int):
    os.makedirs(baseline_folder_output, exist_ok=True)
    keysteps_path = keysteps
    keysteps = json.load(open(keysteps))
    for scenario in keysteps["taxonomy"]:
        print(scenario)
        print()
        tg = baseline_transition_graph(keysteps_path, scenario, max_length)
        tg.save_transition_matrix(os.path.join(baseline_folder_output, f'{scenario}_transition_matrix.csv'))
        G, _ = tg.create_networkx_graph()
        save_graph_as_svg(G, scenario, baseline_folder_output)

if __name__ == '__main__':
    main()