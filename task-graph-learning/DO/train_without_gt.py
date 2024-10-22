# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import numpy as np
import networkx as nx
import torch
import click
import os
import sys

try:
    from taskgraph.task_graph_learning import (DO, 
                                            extract_predecessors,
                                            delete_redundant_edges,
                                            load_config_task_graph_learning,
                                            sequences_accuracy)
except:
    raise Exception("You need to install the TGML library. Please read the README.md file.")

@click.command()
@click.option("--config", "-cfg", type=str, required=True, help="Path to the config file. You can find the config file in the config folder.")
@click.option("--log", "-l", type=bool, default=False, is_flag=True, help="Log the output to a file.")
@click.option("--seed", "-s", type=int, default=None, help="Seed for reproducibility.")
@click.option("--augmentation", "-ag", type=bool, default=False, is_flag=True, help="Augmentation of the sequences.")
@click.option("--device", "-d", type=str, default="cuda:0", help="Device to use.")
@click.option("--relaxed", "-r", type=bool, default=False, is_flag=True, help="Relaxed edges.")
def main(config:str, log:bool, seed:int, augmentation:bool, device:str, relaxed:bool):
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

    # Activity name
    activity_name = cfg.ACTIVITY_NAME

    # Output path
    output_path = cfg.OUTPUT_DIR

    # Check output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Check if log is True
    if log:
        # Redirect the output to the log file
        sys.stdout = open(os.path.join(output_path, "log.txt"), "w")

    # Annotations
    annotations_json = cfg.ANNOTATIONS["annotations"]

    # Taxonomy
    taxonomy_json = cfg.ANNOTATIONS["taxonomy"][activity_name]

    # Number of nodes
    num_nodes = len(taxonomy_json) + 4

    # ID BETA
    id_beta = 0

    # ID START
    id_start = 1

    # ID END
    id_end = num_nodes - 2

    # ID GAMMA
    id_gamma = num_nodes - 1

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

    early_stopping = cfg.EARLY_STOPPING

    # Create the mapping nodes
    mapping_nodes = {}
    for i, node in enumerate(taxonomy_json):
        mapping_nodes[node] = i + 2
    r_mapping_nodes = {v: k for k, v in mapping_nodes.items()}

    # Get train sequences
    all_train_sequences = []

    """
    We consider only the first occurrence of a step in the sequence.
    We do not consider a step if it is already in the sequence.

    Example:
    [1,2,1,3,2,4] -> [1,2,3,4]
    """
    train_sequences_1 = []
    for video in annotations_json:
        if annotations_json[video]["scenario"] != activity_name:
            continue
        sequences = []
        for segment in annotations_json[video]["segments"]:
            step = mapping_nodes[str(segment["step_id"])]
            if step in sequences:
                continue
            sequences.append(step)
        if augmentation:
            train_sequences_1.append(sequences)
        elif sequences not in train_sequences_1:
            train_sequences_1.append(sequences)

    if augmentation:
        """
        We consider the last occurrence of a step in the sequence.
        We remove the step if it is already in the sequence.

        Example:
        [1,2,1,3,2,4] -> [1,3,2,4]
        """
        train_sequences_2 = []
        for video in annotations_json:
            if annotations_json[video]["scenario"] != activity_name:
                continue
            sequences = []
            for segment in annotations_json[video]["segments"]:
                step = mapping_nodes[str(segment["step_id"])]
                if step in sequences:
                    sequences.remove(step)
                sequences.append(step)
            train_sequences_2.append(sequences)
        
        """
        We consider all the occurrences of a step in the sequence.
        Then, during the training, we consider the step only once, and
        we use a random threshold to decide if the step is part of the sequence or not.

        Go in the _loss.py file of the library.
        """
        train_sequences_3 = []
        for video in annotations_json:
            if annotations_json[video]["scenario"] != activity_name:
                continue
            sequences = []
            for segment in annotations_json[video]["segments"]:
                step = mapping_nodes[str(segment["step_id"])]
                sequences.append(step)
            train_sequences_3.append(sequences)

    # Concatenate all the sequences
    all_train_sequences = []
    if augmentation:
        all_train_sequences = [train_sequences_1, train_sequences_2, train_sequences_3]
    else:
        all_train_sequences = [train_sequences_1]

    train_seq = [[([id_beta] + [id_start] + seq + [id_end] + [id_gamma]) for seq in train_sequences] for train_sequences in all_train_sequences]

    train_seq_no_beta_gamma = []
    for train_sequences in all_train_sequences:
        train_seq_no_beta_gamma.extend([([id_start - 1] + [(s - 1) for s in seq] + [id_end - 1]) for seq in train_sequences])

    # Select the device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Create the model
    net = DO(num_nodes, device).to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    # Train the model
    all_y = train_seq
    y = []
    for seq in all_y:
        y.extend(seq)
    net.train()

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
        pred_adjacency_matrix = np.where(mask, pred_adjacency_matrix, 0)
        pred = np.where(mask, pred, 0)

        # Take the matrix removing first row, first column, last row and las column
        pred_adjacency_matrix = pred_adjacency_matrix[1:-1, 1:-1]
        pred = pred[1:-1, 1:-1]
        pred_adjacency_matrix = np.where(pred_adjacency_matrix < (1/(num_nodes-2)), 0, 1)
        GP = nx.DiGraph(pred_adjacency_matrix)

        # Delete cycles when the graph is not a Directed Acyclic Graph (DAG) and epoch > 40
        # We set 40 due to the fact that at the beginning the graph is not a DAG and contains a lot of cycles.
        # If you decide to not use the early stopping, this code will not be executed.
        if not nx.is_directed_acyclic_graph(GP) and i+1 > 40 and early_stopping is not None:
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

        in_degree_zeros = [node for node in GP.nodes if GP.in_degree(node) == 0 and node != id_end-1 and node != id_start-1]
        for node in in_degree_zeros:
            if node not in masked_nodes:
                GP.add_edge(id_end-1, node)

        out_degree_zeros = [node for node in GP.nodes if GP.out_degree(node) == 0 and node != id_end-1 and node != id_start-1]
        for node in out_degree_zeros:
            if node not in masked_nodes:
                GP.add_edge(node, id_start-1)

        if relaxed:
            for node in GP.nodes:
                if len(list(GP.successors(node))) == 2 and 0 in GP.successors(node):
                    for successor in list(GP.successors(node)):
                        if successor != 0:
                            GP.remove_edge(node, successor)

        delete_redundant_edges(GP)
                
        for node in GP.nodes:
            if node == 0:
                GP.nodes[node]["label"] = "START"
                GP.nodes[node]["shape"] = "ellipse"
            elif node == len(GP.nodes) - 1:
                GP.nodes[node]["label"] = "END"
                GP.nodes[node]["shape"] = "ellipse"
            else:
                GP.nodes[node]["label"] = str(taxonomy_json[r_mapping_nodes[node + 1]]["id"]) + "_" + taxonomy_json[r_mapping_nodes[node + 1]]["name"]
                GP.nodes[node]["shape"] = "box"

        for node in masked_nodes:
            GP.remove_node(node+1)

        """
        We calculate the sequences accuracy if the graph is a Directed Acyclic Graph (DAG).
        """
        accuracy_score = 0
        if nx.is_directed_acyclic_graph(GP):
            pred_predecessors = extract_predecessors(GP)
            accuracy_score = sequences_accuracy(train_seq_no_beta_gamma, pred_predecessors)

        print(f"Epoch {i+1}: loss = {loss.item()} | accuracy = {accuracy_score}")

        if early_stopping is not None and accuracy_score > early_stopping:
            break

    print("Training completed")
    torch.save(net.state_dict(), os.path.join(output_path, f"{cfg.OUTPUT_NAME}"))

if __name__ == "__main__":
    main()