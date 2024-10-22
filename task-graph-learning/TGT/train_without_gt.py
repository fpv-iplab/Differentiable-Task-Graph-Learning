# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import numpy as np
import networkx as nx
import torch
import click
import os
import random

try:
    from taskgraph.task_graph_learning import (TGT, 
                                            extract_predecessors,
                                            delete_redundant_edges,
                                            load_config_task_graph_learning,
                                            sequences_accuracy)
except:
    raise Exception("You need to install the TGML library. Please read the README.md file.")


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
    beta_end = 0.55

    # Set seed
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Activity name
    activity_name = cfg.ACTIVITY_NAME

    # Output path
    output_path = cfg.OUTPUT_DIR
    
    # Check output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
    net = TGT(device=device).to(device)

    # Input embeds
    embeddings_path = cfg.EMBEDDINGS
    input_embeds = []
    for embedding in os.listdir(os.path.join(embeddings_path, activity_name)):
        input_embeds.append(torch.load(os.path.join(embeddings_path, activity_name, embedding))['text'].view(-1))
    input_embeds = torch.stack(input_embeds, dim=0).to(device)
    input_embeds = torch.nn.Embedding.from_pretrained(input_embeds, freeze=True)
    encoder_input = torch.nn.Sequential(input_embeds).to(device)
    batch_size = torch.tensor(np.arange(input_embeds.weight.size(0))).to(device)
    input_embeds = encoder_input(batch_size).to(device)
    
    # Create the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.LR)
 
    # Train the model
    all_y = train_seq
    y = []
    for seq in all_y:
        y.extend(seq)
    net.train()
    
    for i in range(epochs):
        
        # warm up the beta parameter
        if (i+1) % 200 == 0:
            beta = np.interp(i+1, [0, epochs], [beta, beta_end])

        net.train()
        optimizer.zero_grad()
        loss1, loss2 = net(input_embeds, y, beta=beta)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            net.eval()
            pred_adjacency_matrix, _ = net.get_adjacency_matrix(input_embeds, mask)
            pred_adjacency_matrix = pred_adjacency_matrix.cpu().numpy()
            pred, _ = net.get_adjacency_matrix(input_embeds, mask)
            pred = pred.cpu().numpy()
            
        # Take the matrix removing first row, first column, last row and las column
        pred_adjacency_matrix = pred_adjacency_matrix[1:-1, 1:-1]
        pred = pred[1:-1, 1:-1]
        
        # Set to 0 the values below the threshold
        pred_adjacency_matrix = np.where(pred_adjacency_matrix < (1/(num_nodes-2)), 0, 1)
        GP = nx.DiGraph(pred_adjacency_matrix)
        
        # Delete cycles when the graph is not a Directed Acyclic Graph (DAG) and conditions are met
        if not nx.is_directed_acyclic_graph(GP) and (num_nodes-4) < 40 and (i+1) > (num_nodes-4) * 10:
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
            if node-1 not in masked_nodes:
                GP.add_edge(id_end-1, node)

        out_degree_zeros = [node for node in GP.nodes if GP.out_degree(node) == 0 and node != id_end-1 and node != id_start-1]
        for node in out_degree_zeros:
            if node-1 not in masked_nodes:
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
        accuracy_of_sequences_score = 0
        if nx.is_directed_acyclic_graph(GP):
            pred_predecessors = extract_predecessors(GP)
            accuracy_of_sequences_score = sequences_accuracy(train_seq_no_beta_gamma, pred_predecessors)
        
        print(f"Epoch {i+1}: beta = {beta} | loss1 = {loss1.item()} | loss2 = {loss2.item()} | accuracy of sequences = {accuracy_of_sequences_score}")
        
        """
        In this case we stop the training when the accuracy of the sequences is between 0.70 and 0.73.
        We want a graph that is not too complex and not too simple.
        """
        if accuracy_of_sequences_score > 0.70 and accuracy_of_sequences_score < 0.73:
            break
    
    print("Training completed")
    torch.save(net.state_dict(), os.path.join(output_path, f"model_{activity_name}.pth"))

if __name__ == "__main__":
    main()