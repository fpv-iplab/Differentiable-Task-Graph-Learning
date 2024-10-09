# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import networkx as nx
import click
import os
import json
import threading

from sklearn.metrics import precision_score as precision, recall_score as recall, f1_score

try:
    from taskgraph.task_graph_learning import load_config_evaluation_baseline
except:
    raise Exception("Taskgraph library not found. Please install it! Read the README.md file for more information.")

def processing(GT, GP, scenario, output_path):
    # Compute the precision, recall and F1 score
    GT_flatten_adj = nx.adjacency_matrix(GT, nodelist=sorted(GT.nodes)).todense().flatten()
    GP_flatten_adj = nx.adjacency_matrix(GP, nodelist=sorted(GP.nodes)).todense().flatten()
    precision_score = precision(GT_flatten_adj, GP_flatten_adj)
    recall_score = recall(GT_flatten_adj, GP_flatten_adj)
    f1 = f1_score(GT_flatten_adj, GP_flatten_adj)
    print()
    print(f"Scenario: {scenario} | Precision: {precision_score} | Recall: {recall_score} | F1: {f1}")
    out = {
        "Precision": precision_score,
        "Recall": recall_score,
        "F1": f1
    }
    with open(os.path.join(output_path, f"{scenario}.json"), "w") as f:
        json.dump(out, f)

@click.command()
@click.option("--config", "-cfg", type=str, help="Path to the config file. You can find the config file in the config folder.", required=True)
def main(config:str):
    # Load config
    cfg = load_config_evaluation_baseline(config)

    output_path = cfg.OUTPUT_PATH
    # Check output path
    if not os.path.exists(os.path.join(output_path)):
        os.makedirs(os.path.join(output_path))

    gt_graph = cfg.GT_TASK_GRAPHS
    pred_graph = cfg.PRED_TASK_GRAPHS

    gt_graph = sorted(gt_graph)
    pred_graph = sorted(pred_graph)

    threads = []

    for gt, pred in zip(gt_graph, pred_graph):
        # Extract the scenario
        scenario = gt.split('/')[-1].split('.')[0]

        # Read the GT task graph
        task_graph_json = json.load(open(gt))

        # Create NetworkX graph
        GT = nx.DiGraph()

        for node in task_graph_json["steps"]:
            GT.add_node(int(node), label=task_graph_json["steps"][node], shape="box")
        
        for edge in task_graph_json["edges"]:
            GT.add_edge(edge[1], edge[0])

        # Taxonomy
        taxonomy_json = cfg.ANNOTATIONS["taxonomy"]

        # step_idx_description
        description_to_id = {}
        for step_id in taxonomy_json[scenario]:
            description_to_id[taxonomy_json[scenario][step_id]["name"]] = int(step_id)
        mapping_nodes = {}
        for node in GT.nodes:
            if GT.nodes[node]["label"] in description_to_id:
                mapping_nodes[node] = int(description_to_id[GT.nodes[node]["label"]])
                    
        # Read the predicted task graph
        pred = nx.read_multiline_adjlist(pred, create_using=nx.DiGraph, delimiter=";")

        # Take the reverse of the task graph json
        reverse_task_graph_json = {v.lower(): int(k) for k, v in task_graph_json["steps"].items()}

        # Relabel the nodes of the predicted task graph as the nodes of the GT task graph
        dictionary = {}
        for node in pred.nodes:
            if node == "START":
                dictionary[node] = 0
            elif node == "END":
                dictionary[node] = len(pred.nodes) - 1
            else:
                try:
                    dictionary[node] = reverse_task_graph_json[node.split("_")[1].lower()]
                except:
                    dictionary[node] = reverse_task_graph_json[node.lower()]
        pred = nx.relabel_nodes(pred, dictionary)

        for node in pred.nodes:
            pred.nodes[node]["label"] = task_graph_json["steps"][str(node).lower()]
            pred.nodes[node]["shape"] = "box"

        # Start the processing
        print(f"Processing {scenario}")
        thread = threading.Thread(target=processing, args=(GT, pred, scenario, output_path))
        thread.start()
        thread.join()
        threads.append(thread)

    for thread in threads:
        thread.join()
    
if __name__ == "__main__":
    main()