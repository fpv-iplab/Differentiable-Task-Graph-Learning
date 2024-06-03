# This script trains the model on the task graph learning task. The model is trained on the CaptainCook4D dataset.

import numpy as np
import networkx as nx
import torch
import click
import os
import json
import sys

from taskgraph.task_graph_learning import (DO, 
                                           extract_predecessors,
                                           delete_redundant_edges,
                                           load_config_task_graph_learning,
                                           sequences_accuracy)

from sklearn.metrics import precision_score as precision, recall_score as recall, f1_score

@click.command()
@click.option("--config", "-cfg", type=str, required=True, help="Path to the config file. You can find the config file in the config folder.")
@click.option("--log", "-l", type=bool, default=False, is_flag=True, help="Log the output to a file.")
@click.option("--seed", "-s", type=int, default=None, help="Seed for reproducibility.")
def main(config:str, log:bool, seed:int):
    # Load config
    cfg = load_config_task_graph_learning(config)

    # Epochs
    epochs = cfg.EPOCHS

    # Beta
    beta = cfg.BETA

    # Set seed
    if seed is not None:
        cfg.SEED = seed
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    activity_name = cfg.ACTIVITY_NAME

    # Output path
    output_path = cfg.OUTPUT_DIR

    # Check output path
    if not os.path.exists(os.path.join(output_path, f"{activity_name}")):
        os.makedirs(os.path.join(output_path, f"{activity_name}"))
    
    # Check if log is True
    if log:
        # Redirect the output to the log file
        sys.stdout = open(os.path.join(output_path, f"{activity_name}", "log.txt"), "w")

    # Load the task graphs
    task_graph = os.path.join(cfg.TASK_GRAPHS, f"{activity_name}.json")
    task_graph_json = json.load(open(task_graph))

    # Create NetworkX graph
    GT = nx.DiGraph()

    for node in task_graph_json["steps"]:
        GT.add_node(int(node), label=task_graph_json["steps"][node], shape="box")
    
    for edge in task_graph_json["edges"]:
        GT.add_edge(edge[1], edge[0])

    """
    ID beta and ID gamma are two distinct token nodes, separate from the standard start and end tokens, 
    that are prepended to the input sequences along with the start and end tokens. 
    The purpose of including these additional tokens is to enhance the model's ability to accurately learn and identify the positions of the "start" and "end" 
    nodes within the generated task graph.

    To implement this, we introduce two additional nodes into the input sequences: ID beta is added at the beginning and ID gamma at the end of each sequence. 
    This placement helps in clearly defining the sequence boundaries for the model during training.

    Once the task graph is generated from these sequences, we systematically remove the rows and columns corresponding to ID beta and ID gamma in the final 
    task graph representation. This step ensures that these auxiliary nodes do not interfere with the interpretation and application of the task graph, 
    focusing only on the essential elements of the sequence.
    """

    # ID beta
    id_beta = 0

    # ID START
    id_start = 1

    # ID END
    id_end = len(GT.nodes)

    # ID gamma
    id_gamma = len(GT.nodes) + 1

    # Pygraphviz
    A = nx.nx_agraph.to_agraph(GT)
    A.layout('dot')
    A.draw(os.path.join(output_path, f"{activity_name}", "GT_graph.png"))

    # Number of nodes
    num_nodes = len(GT.nodes) + 2

    # Annotations
    annotations_json = cfg.ANNOTATIONS["annotations"]

    # Taxonomy
    taxonomy_json = cfg.ANNOTATIONS["taxonomy"]

    # step_idx_description
    description_to_id = {}
    for step_id in taxonomy_json[activity_name]:
        description_to_id[taxonomy_json[activity_name][step_id]["name"]] = int(step_id)
    mapping_nodes = {}
    for node in GT.nodes:
        if GT.nodes[node]["label"] in description_to_id:
            mapping_nodes[node] = int(description_to_id[GT.nodes[node]["label"]])
                
    id_to_node = {mapping_nodes[node]: node for node in mapping_nodes}

    # Get CaptainCook4D sequences
    train_sequences = []
    for video in annotations_json:
        if annotations_json[video]["scenario"] != activity_name:
            continue
        sequences = []
        for segment in annotations_json[video]["segments"]:
            step = segment["step_id"]
            if step in sequences:
                continue
            sequences.append(step)
        if sequences not in train_sequences:
            train_sequences.append(sequences)
    
    # Convert the sequences to node ids and add the beta, start, end and gamma nodes
    for i, seq in enumerate(train_sequences):
        train_sequences[i] = [(id_to_node[id_step] + 1) for id_step in seq]
    train_seq = np.array([([id_beta] + [id_start] + seq + [id_end] + [id_gamma]) for seq in train_sequences])
    train_seq_no_beta_gamma = np.array([([id_start - 1] + [(s - 1) for s in seq] + [id_end - 1]) for seq in train_sequences])

    # Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the model
    net = DO(num_nodes, device).to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    # Train the model
    y = train_seq
    net.train()
    best_accuracy_of_sequences = -float('inf')
    early_stopping_counter = 0

    for i in range(epochs):
        net.train()
        optimizer.zero_grad()
        loss = net(y, beta=beta)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            net.eval()
            pred_adjacency_matrix = net.get_adjacency_matrix().cpu().numpy()
            pred = net.get_adjacency_matrix().cpu().numpy()
        
        # Take the matrix removing first row, first column, last row and las column
        pred_adjacency_matrix = pred_adjacency_matrix[1:-1, 1:-1]
        pred = pred[1:-1, 1:-1]
        pred_adjacency_matrix = np.where(pred_adjacency_matrix < (1/(num_nodes-2)), 0, 1)
        GP = nx.DiGraph(pred_adjacency_matrix)

        in_degree_zeros = [node for node in GP.nodes if GP.in_degree(node) == 0 and node != id_end-1 and node != id_start-1]
        for node in in_degree_zeros:
            GP.add_edge(id_end-1, node)

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
        pred_predecessors = extract_predecessors(GP)
        """
        We calculate the sequences accuracy if the graph is a Directed Acyclic Graph (DAG).
        """
        sequences_accuracy_score = 0
        if nx.is_directed_acyclic_graph(GP):
            sequences_accuracy_score = sequences_accuracy(train_seq_no_beta_gamma, pred_predecessors)
            if sequences_accuracy_score > 0.95:
                if sequences_accuracy_score == 1 and sequences_accuracy_score == best_accuracy_of_sequences:
                    early_stopping_counter += 1
                elif sequences_accuracy_score >= best_accuracy_of_sequences:
                    best_accuracy_of_sequences = sequences_accuracy_score
                    early_stopping_counter = 0
                    A = nx.nx_agraph.to_agraph(GP)
                    A.layout('dot')
                    A.draw(os.path.join(output_path, f"{activity_name}", "GP_graph_best.png"))
                else:
                    early_stopping_counter += 1
        
        if early_stopping_counter == 25:
            break

        print(f"Epoch {i+1}: beta = {beta} | loss = {loss.item()} | precision = {precision_score} | recall = {recall_score} | accuracy of sequences = {sequences_accuracy_score}")
    
    print("Training completed")
    torch.save(net.state_dict(), os.path.join(output_path, f"{activity_name}", f"model_{activity_name}_final.pth"))

    A = nx.nx_agraph.to_agraph(GP)
    A.layout('dot')
    A.draw(os.path.join(output_path, f"{activity_name}", "GP_graph.png"))

    # Output the results
    results = {
        "Precision": precision_score,
        "Recall": recall_score,
        "F1": f1,
        "Accuracy of sequences": sequences_accuracy_score
    }

    with open(os.path.join(output_path, f"{activity_name}", f"{cfg.SEED}_results.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()