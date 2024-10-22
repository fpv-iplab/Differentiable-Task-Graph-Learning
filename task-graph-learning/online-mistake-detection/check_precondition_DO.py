# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import click
import os
import networkx as nx
import json
import torch
import numpy as np

try:
    from taskgraph.mistake_detection import check_precondition
    from taskgraph.task_graph_learning import DO, delete_redundant_edges, load_config_mistake_detection
except:
    raise Exception("You need to install the TGML library. Please read the README.md file.")

from sklearn.metrics import precision_score, recall_score, f1_score

@click.command()
@click.option("--config", "-cfg", type=str, required=True, help="Path to the configuration file.")
@click.option("--relaxed", "-r", type=bool, default=False, is_flag=True, help="Relaxed edges.")
def main(config:str, relaxed:bool):
    cfg = load_config_mistake_detection(config)
    
    # Load the configuration file
    keysteps = cfg.ANNOTATIONS
    pred_graph = cfg.PRED_TASK_GRAPHS
    scenario_name = cfg.SCENARIO_NAME
    folder_output = cfg.OUTPUT_PATH
    taxonomy = keysteps["taxonomy"]
    os.makedirs(folder_output, exist_ok=True)
    
    # Create the results dictionary
    results_scenario = {}
    results_scenario[scenario_name] = {}

    # Load the predicted graph
    pred_graph = torch.load(pred_graph)
    num_nodes = pred_graph["adjacency_matrix.task_graph"].shape[0]
    id_start = 1
    id_end = num_nodes - 2
    pred_graph_ = DO(num_nodes)
    pred_graph_.load_state_dict(pred_graph)
    pred_graph = pred_graph_

    # Masked nodes
    masked_nodes = cfg.MASK
    if masked_nodes is None:
        masked_nodes = []
    else:
        masked_nodes = [int(node) for node in masked_nodes.split(",")]
    mask = np.ones((num_nodes, num_nodes), dtype=bool)
    for node in masked_nodes:
        mask[node+2, :] = False
        mask[:, node+2] = False

    pred_graph.eval()
    with torch.no_grad():
        pred_adjacency_matrix = pred_graph.get_adjacency_matrix().cpu().numpy()
        pred = pred_graph.get_adjacency_matrix().cpu().numpy()
    pred_adjacency_matrix = np.where(mask, pred_adjacency_matrix, 0)
    pred = np.where(mask, pred, 0)

    # Take the matrix removing first row, first column, last row and las column
    pred_adjacency_matrix = pred_adjacency_matrix[1:-1, 1:-1]
    pred = pred[1:-1, 1:-1]

    # Set to 0 the values below the threshold
    pred_adjacency_matrix = np.where(pred_adjacency_matrix < (1/(num_nodes-2)), 0, 1)
    GP = nx.DiGraph(pred_adjacency_matrix)

    # Check if the graph is a Directed Acyclic Graph (DAG)
    # If not, remove the edge with the lowest weight in the cycle
    if not nx.is_directed_acyclic_graph(GP):
        for cycle in nx.simple_cycles(GP):
            # Find the edge with the lowest weight
            min_weight = np.inf
            min_edge = None
            for i in range(len(cycle) - 1):
                if pred[cycle[i], cycle[i+1]] < min_weight:
                    min_weight = pred[cycle[i], cycle[i+1]]
                    min_edge = (cycle[i], cycle[i+1])
            if min_edge is not None and GP.has_edge(min_edge[0], min_edge[1]):
                GP.remove_edge(min_edge[0], min_edge[1])

    if relaxed:
        for node in GP.nodes:
            if len(list(GP.successors(node))) == 2 and 0 in GP.successors(node):
                for successor in list(GP.successors(node)):
                    if successor != 0:
                        GP.remove_edge(node, successor)
    
    in_degree_zeros = [node for node in GP.nodes if GP.in_degree(node) == 0 and node != id_end-1 and node != id_start-1]
    for node in in_degree_zeros:
        if node-1 not in masked_nodes:
            GP.add_edge(id_end-1, node)
    
    out_degree_zeros = [node for node in GP.nodes if GP.out_degree(node) == 0 and node != id_end-1 and node != id_start-1]
    for node in out_degree_zeros:
        if node-1 not in masked_nodes:
            GP.add_edge(node, id_start-1)

    delete_redundant_edges(GP)

    taxonomy_json = taxonomy[scenario_name]
    mappings = {
        0: "START",
        len(GP.nodes) - 1: "END"
    }
    for i, step in enumerate(taxonomy_json):
        mappings[i + 1] = step + "_" + taxonomy_json[step]["name"]

    for node in GP.nodes:
        if node == 0:
            GP.nodes[node]["label"] = "START"
            GP.nodes[node]["shape"] = "ellipse"
        elif node == len(GP.nodes) - 1:
            GP.nodes[node]["label"] = "END"
            GP.nodes[node]["shape"] = "ellipse"
        else:
            GP.nodes[node]["label"] = mappings[node]
            GP.nodes[node]["shape"] = "box"

    pred_graph = GP

    # Delete masked nodes
    for node in masked_nodes:
        pred_graph.remove_node(node+1)

    nx.relabel_nodes(pred_graph, mappings, copy=False)
    G_pg = nx.nx_agraph.to_agraph(pred_graph)
    G_pg.layout(prog='dot')
    G_pg.draw(os.path.join(folder_output, f"{scenario_name}.png"))

    # Chenge orientation of the edges of the graph
    for edge in list(pred_graph.edges):
        pred_graph.remove_edge(edge[0], edge[1])
        pred_graph.add_edge(edge[1], edge[0])

    # Calculate the anomaly of the sequences
    gts = []
    preds = []
    mistake_results = {}
    for video in keysteps["annotations"]:
        if keysteps["annotations"][video]["scenario"] == scenario_name:
            mistake_results[video] = []
            segments = keysteps["annotations"][video]["segments"]
            results, predictions = check_precondition(pred_graph, segments, video)
            mistake_results[video].append(results)
            gt = [0] * (len(predictions) - 1) + [1]
            gts.extend(gt)
            preds.extend(predictions)
            print(predictions)
    
    json.dump(mistake_results, open(os.path.join(folder_output, f"mistake_results_{scenario_name}.json"), "w"), indent=4)
    results_scenario[scenario_name]["precision_correct"], results_scenario[scenario_name]["precision_mistake"] = precision_score(preds, gts, average=None)
    results_scenario[scenario_name]["precision_mean"] = precision_score(preds, gts, average="macro")
    results_scenario[scenario_name]["recall_correct"], results_scenario[scenario_name]["recall_mistake"] = recall_score(preds, gts, average=None)
    results_scenario[scenario_name]["recall_mean"] = recall_score(preds, gts, average="macro")
    results_scenario[scenario_name]["f1_correct"], results_scenario[scenario_name]["f1_mistake"] = f1_score(preds, gts, average=None)
    results_scenario[scenario_name]["f1_mean"] = f1_score(preds, gts, average="macro")

    print()
    print(results_scenario)
    print()
    
    # Save the results in JSON and CSV format
    json.dump(results_scenario, open(os.path.join(folder_output, "results.json"), "w"))


if __name__ == '__main__':
    main()