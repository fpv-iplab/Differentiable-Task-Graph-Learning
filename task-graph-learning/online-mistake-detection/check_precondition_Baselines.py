# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import networkx as nx
import click
import os
import json

try:
    from taskgraph.task_graph_learning import load_config_mistake_detection
    from taskgraph.mistake_detection import check_precondition
except:
    raise Exception("You need to install the TGML library. Please read the README.md file.")

from sklearn.metrics import precision_score, recall_score, f1_score

@click.command()
@click.option("--config", "-cfg", type=str, required=True, help="Path to the configuration file.")
def main(config:str):
    cfg = load_config_mistake_detection(config)
    
    # Load the configuration file
    keysteps = cfg.ANNOTATIONS
    pred_graph = cfg.PRED_TASK_GRAPHS
    pred_graph_name = cfg.SCENARIO_NAME
    folder_output = cfg.OUTPUT_PATH
    os.makedirs(folder_output, exist_ok=True)

    # Load the predicted graph
    pred_graph = nx.read_multiline_adjlist(pred_graph, create_using=nx.DiGraph, delimiter=";")
    G_pg = nx.nx_agraph.to_agraph(pred_graph)
    G_pg.layout(prog='dot')
    G_pg.draw(os.path.join(folder_output, f"{pred_graph_name}.png"))

    # Change the orientation of the edges of the graph for simplicity on the usage of the check_precondition function
    for edge in list(pred_graph.edges):
        pred_graph.remove_edge(edge[0], edge[1])
        pred_graph.add_edge(edge[1], edge[0])

    # Create the results dictionary
    results_scenario = {}
    results_scenario[pred_graph_name] = {}

    # Find mistakes in the scenario
    gts = []
    preds = []
    mistake_results = {}
    for video in keysteps["annotations"]:
        mistake_results[video] = []
        if keysteps["annotations"][video]["scenario"] == pred_graph_name:
            segments = keysteps["annotations"][video]["segments"]
            results, predictions = check_precondition(pred_graph, segments, video)
            mistake_results[video].append(results)
            gt = [0] * (len(predictions) - 1) + [1]
            gts.extend(gt)
            preds.extend(predictions)
            print(predictions)

    # Save the mistake information in JSON format
    json.dump(mistake_results, open(os.path.join(folder_output, f"mistake_results_{pred_graph_name}.json"), "w"), indent=4)
    
    # Calculate the precision correct, precision mistake, recall correct, recall mistake, f1 correct, f1 mistake, precision mean, recall mean, and f1 mean
    results_scenario[pred_graph_name]["precision_correct"], results_scenario[pred_graph_name]["precision_mistake"] = precision_score(preds, gts, average=None)
    results_scenario[pred_graph_name]["precision_mean"] = precision_score(preds, gts, average="macro")
    results_scenario[pred_graph_name]["recall_correct"], results_scenario[pred_graph_name]["recall_mistake"] = recall_score(preds, gts, average=None)
    results_scenario[pred_graph_name]["recall_mean"] = recall_score(preds, gts, average="macro")
    results_scenario[pred_graph_name]["f1_correct"], results_scenario[pred_graph_name]["f1_mistake"] = f1_score(preds, gts, average=None)
    results_scenario[pred_graph_name]["f1_mean"] = f1_score(preds, gts, average="macro")

    print()
    print(results_scenario)
    print()
    
    # Save the results in JSON and CSV format
    json.dump(results_scenario, open(os.path.join(folder_output, "results.json"), "w"))


if __name__ == '__main__':
    main()