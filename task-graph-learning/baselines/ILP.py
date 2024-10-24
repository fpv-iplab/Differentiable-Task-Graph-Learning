# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

"""
This script generates a baseline ILP (Integer Linear Programming) solution for a given task graph dataset.
The ILP solution is based on the provided keysteps file and objective (ACCURACY or PRECISION).
The resulting ILP solution is saved as an SVG file in the specified baseline folder output.
"""

import json
import click
import os
import networkx as nx
import numpy as np
import random

try:
    from taskgraph.task_graph_learning import baseline_ILP, save_graph_as_svg
except:
    raise Exception("You need to install the TGML library. Please read the README.md file.")

@click.command()
@click.option('--keysteps', help='File JSON containing the keysteps', required=True)
@click.option('--objective', help='Objective: ACCURACY | PRECISION', required=True, default='acc')
@click.option('--baseline_folder_output', '-bfo', help='Baseline folder output', required=True)
@click.option('--augmentations', '-a', help='Use augmentations', is_flag=True, default=False)
@click.option('--max_length', '-ml', help='Max length of the sequences', default=-1)
def main(keysteps:str, objective:str, baseline_folder_output:str, augmentations:bool, max_length:int):
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Create the baseline folder output
    os.makedirs(baseline_folder_output, exist_ok=True)
    
    # Open the keysteps file
    keysteps = json.load(open(keysteps))
    
    # Create the scenarios_sequences dictionary
    scenarios_sequences = {}
    for scenario in keysteps["taxonomy"]:
        print(scenario)
        print()
        scenarios_sequences[scenario] = {
            "sequences": []
        }
        
        # If max_length is -1, we set it to the length of all videos + 1
        if max_length == -1:
            max_length = len(keysteps["annotations"]) + 1
        
        for video in keysteps["annotations"]:
            if keysteps["annotations"][video]["scenario"] != scenario:
                continue
            
            # We create three different sequences for each video if augmentations is True
            sequence1 = []
            sequence2 = []
            sequence3 = []
            
            # We add the steps to the sequences
            for segment in keysteps["annotations"][video]["segments"]:
                step_id = segment["step_id"]
                if step_id not in sequence1:
                    sequence1.append(step_id)
                if augmentations:
                    if step_id in sequence2:
                        sequence2.remove(step_id)
                        sequence2.append(step_id)
                    sequence3.append(step_id)
            
            # We add the sequences to the scenarios_sequences dictionary if the length is less than max_length
            if len(scenarios_sequences[scenario]["sequences"]) < max_length:
                if max_length != len(keysteps["annotations"]) + 1:
                    scenarios_sequences[scenario]["sequences"].append(sequence1)
                elif sequence1 not in scenarios_sequences[scenario]["sequences"]:
                    scenarios_sequences[scenario]["sequences"].append(sequence1)
                if augmentations:
                    if sequence2 not in scenarios_sequences[scenario]["sequences"]:
                        scenarios_sequences[scenario]["sequences"].append(sequence2)
                    if sequence3 not in scenarios_sequences[scenario]["sequences"]:
                        scenarios_sequences[scenario]["sequences"].append(sequence3)
        if max_length != len(keysteps["annotations"]) + 1:
            while len(scenarios_sequences[scenario]["sequences"]) < max_length:
                scenarios_sequences[scenario]["sequences"].append(random.choice(scenarios_sequences[scenario]["sequences"]))
        threshold = 1 / len(keysteps["taxonomy"][scenario])
        _, _, graph, _, _ = baseline_ILP(scenarios_sequences[scenario]["sequences"], objective, thresh=threshold)
        mapping_to_taxonomy = {}
        for node in graph.nodes:
            if node == "START":
                mapping_to_taxonomy[node] = "START"
            elif node == "END":
                mapping_to_taxonomy[node] = "END"
            else:
                mapping_to_taxonomy[node] = str(node) + "_" + keysteps["taxonomy"][scenario][str(node)]["name"]
        graph = nx.relabel_nodes(graph, mapping_to_taxonomy, copy=False)
        save_graph_as_svg(graph, scenario, baseline_folder_output)

if __name__ == '__main__':
    main()