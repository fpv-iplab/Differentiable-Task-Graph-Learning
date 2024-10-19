# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import numpy as np
import networkx as nx
import torch
import click
import os
import json
import random

try:
    from taskgraph.task_graph_learning import (TGT, 
                                            delete_redundant_edges,
                                            load_config_task_graph_learning
                                            )   
except:
    raise Exception("You need to install the TGML library. Please read the README.md file.")

from sklearn.metrics import precision_score as precision, recall_score as recall, f1_score

# Set the environment variable for CUBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

@click.command()
@click.option("--config", "-cfg", type=str, required=True, help="Path to the config file. You can find the config file in the config folder.")
@click.option("--pre_trained", type=str, required=True, help="Path to the pre-trained model.")
@click.option("--device", "-d", type=str, default="cuda:0", help="Device to use for training.")
def main(config:str, pre_trained:str, device:str):

    # Load config
    cfg = load_config_task_graph_learning(config)
    
    # Set seed
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    
    # Set the deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Activity name (e.g., "coffee")
    activity_name = cfg.ACTIVITY_NAME

    # Output path
    output_path = cfg.OUTPUT_DIR + "-unified"

    # Check output path
    if not os.path.exists(os.path.join(output_path, f"{activity_name}")):
        os.makedirs(os.path.join(output_path, f"{activity_name}"))

    # Load the task graphs
    task_graph = os.path.join(cfg.TASK_GRAPHS, f"{activity_name}.json")
    task_graph_json = json.load(open(task_graph))

    # Create NetworkX graph
    GT = nx.DiGraph()

    for node in task_graph_json["steps"]:
        GT.add_node(int(node), label=task_graph_json["steps"][node], shape="box")
    
    for edge in task_graph_json["edges"]:
        GT.add_edge(edge[1], edge[0])

    # ID START
    id_start = 1

    # ID END
    id_end = len(GT.nodes)

    # Pygraphviz
    A = nx.nx_agraph.to_agraph(GT)
    A.layout('dot')
    A.draw(os.path.join(output_path, f"{activity_name}", "GT_graph.png"))

    # Number of nodes
    num_nodes = len(GT.nodes) + 2

    # Select the device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Create the model
    net = TGT(d_model=4096, device=device, dropout=0.25).to(device)

    # Input embeds
    embeddings_path = cfg.EMBEDDINGS
    input_embeds = []
    for embedding in os.listdir(os.path.join(embeddings_path, activity_name)):
        input_embeds.append(torch.load(os.path.join(embeddings_path, activity_name, embedding))['text'].view(-1))
    input_embeds = torch.stack(input_embeds, dim=0)
    input_embeds = torch.nn.Embedding.from_pretrained(input_embeds, freeze=True)
    encoder_input = torch.nn.Sequential(input_embeds).to(device)
    batch_size = torch.tensor(np.arange(input_embeds.weight.size(0))).to(device)

    if pre_trained is not None:
        net.load_state_dict(torch.load(pre_trained))
        print(f"Pre-trained model loaded from {pre_trained}")
    else:
        print("No pre-trained model loaded")
        raise ValueError("No pre-trained model loaded")
 
    # Use the model to predict the adjacency matrix
    with torch.no_grad():
        net.eval()
        pred_adjacency_matrix, _ = net.get_adjacency_matrix(encoder_input(batch_size))
        pred_adjacency_matrix = pred_adjacency_matrix.cpu().numpy()
        pred, _ = net.get_adjacency_matrix(encoder_input(batch_size))
        pred = pred.cpu().numpy()
        
    # Take the matrix removing first row, first column, last row and las column
    pred_adjacency_matrix = pred_adjacency_matrix[1:-1, 1:-1]
    pred = pred[1:-1, 1:-1]

    # Threshold the matrix
    pred_adjacency_matrix = np.where(pred_adjacency_matrix < (1/(num_nodes-2)), 0, 1)
    GP = nx.DiGraph(pred_adjacency_matrix)
    
    # Check if the graph is a Directed Acyclic Graph (DAG)
    # If not, remove the edge with the lowest weight in the cycle
    if not nx.is_directed_acyclic_graph(GP):
        for cycle in nx.simple_cycles(GP):
            # Find the edge with the lowest weight
            min_weight = np.inf
            min_edge = None
            for j in range(len(cycle) - 1):
                if pred[cycle[j], cycle[j+1]] < min_weight:
                    min_weight = pred[cycle[j], cycle[j+1]]
                    min_edge = (cycle[j], cycle[j+1])
            if min_edge is not None and GP.has_edge(min_edge[0], min_edge[1]):
                GP.remove_edge(min_edge[0], min_edge[1])

    # Take all nodes with in_degree = 0 and add an edge from the end node to the node
    in_degree_zeros = [node for node in GP.nodes if GP.in_degree(node) == 0 and node != id_end-1 and node != id_start-1]
    for node in in_degree_zeros:
        GP.add_edge(id_end-1, node)

    # Take all nodes with out_degree = 0 and add an edge from the node to the start node
    out_degree_zeros = [node for node in GP.nodes if GP.out_degree(node) == 0 and node != id_end-1 and node != id_start-1]
    for node in out_degree_zeros:
        GP.add_edge(node, id_start-1)

    delete_redundant_edges(GP)

    for node in GP.nodes:
        GP.nodes[node]["label"] = GT.nodes[node]["label"]
        GP.nodes[node]["shape"] = "box"

    GT_flatten_adj = nx.adjacency_matrix(GT, nodelist=sorted(GT.nodes)).todense().flatten()
    GP_flatten_adj = nx.adjacency_matrix(GP, nodelist=sorted(GP.nodes)).todense().flatten()
    precision_score = precision(GT_flatten_adj, GP_flatten_adj)
    recall_score = recall(GT_flatten_adj, GP_flatten_adj)
    f1 = f1_score(GT_flatten_adj, GP_flatten_adj)

    A = nx.nx_agraph.to_agraph(GP)
    A.layout('dot')
    A.draw(os.path.join(output_path, f"{activity_name}", "GP_graph_all.png"))

    output_results = {
        "Precision": precision_score,
        "Recall": recall_score,
        "F1": f1
    }

    with open(os.path.join(output_path, f"{activity_name}", "one_model_results.json"), "w") as f:
        json.dump(output_results, f, indent=4)

if __name__ == "__main__":
    main()