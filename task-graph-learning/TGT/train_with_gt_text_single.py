# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import numpy as np
import networkx as nx
import torch
import click
import os
import json
import sys
import wandb
import random

try:
    from taskgraph.task_graph_learning import (TGT, 
                                            extract_predecessors,
                                            delete_redundant_edges,
                                            load_config_task_graph_learning,
                                            sequences_accuracy)
except:
    raise Exception("You need to install the TGML library. Please read the README.md file.")

from sklearn.metrics import precision_score as precision, recall_score as recall, f1_score

# Set the environment variable for CUBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

@click.command()
@click.option("--config", "-cfg", type=str, required=True, help="Path to the config file. You can find the config file in the config folder.")
@click.option("--pre_trained", type=str, default=None, help="Path to the pre-trained model.")
@click.option("--log", "-l", type=bool, default=False, is_flag=True, help="Log the output in a file.")
@click.option("--seed", "-s", type=int, default=42, help="Seed for reproducibility.")
@click.option("--cuda", type=int, default=0, help="CUDA device to use.")
@click.option("-w", type=bool, default=False, is_flag=True, help="Use wandb.")
@click.option("--project_name", type=str, default="TGT", help="Project name for wandb.")
@click.option("--entity", type=str, default=None, help="Entity name for wandb.")
@click.option("--save", type=bool, default=False, is_flag=True, help="Save the model.")
def main(config:str, pre_trained:str, log:bool, seed:int, cuda:int, w:bool, project_name:str, entity:str, save:bool):
    # Check if wandb is True
    if w:
        if entity is None:
            raise ValueError("You need to specify the entity name.")
        # Initialize wandb
        wandb.login()
        wandb.init(project=project_name, entity=entity)
    
    # Load config
    cfg = load_config_task_graph_learning(config)

    # Activity name (e.g., "coffee")
    activity_name = cfg.ACTIVITY_NAME

    # Epochs
    epochs = cfg.EPOCHS

    # Beta and beta_end
    beta = 1.0
    beta_end = 0.05

    # Set seed
    cfg.SEED = seed
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    # Set the deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

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

    # Add nodes and edges to the graph
    for node in task_graph_json["steps"]:
        GT.add_node(int(node), label=task_graph_json["steps"][node], shape="box")
    # The edges are reversed because the task graph is in the opposite direction (read the paper for more information)
    for edge in task_graph_json["edges"]:
        GT.add_edge(edge[1], edge[0])

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

    # Get the mapping from the description to the id of the keysteps inside the taxonomy
    description_to_id = {}
    for step_id in taxonomy_json[activity_name]:
        description_to_id[taxonomy_json[activity_name][step_id]["name"]] = int(step_id)
        
    # Create the mapping from the node to the id of the keystep
    mapping_nodes = {}
    for node in GT.nodes:
        if GT.nodes[node]["label"] in description_to_id:
            mapping_nodes[node] = int(description_to_id[GT.nodes[node]["label"]])
    
    # Create the mapping from the id of the keystep to the node
    id_to_node = {mapping_nodes[node]: node for node in mapping_nodes}

    # Get train sequences
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
        train_sequences.append(sequences)
    
    # Convert the sequences to node ids and add the beta, start, end and gamma nodes
    for i, seq in enumerate(train_sequences):
        train_sequences[i] = [(id_to_node[id_step] + 1) for id_step in seq]
    train_seq = [([id_beta] + [id_start] + seq + [id_end] + [id_gamma]) for seq in train_sequences]
    train_seq_no_beta_gamma = [([id_start - 1] + [(s - 1) for s in seq] + [id_end - 1]) for seq in train_sequences]

    # Select the device
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    # Create the model
    net = TGT(d_model=4096, device=device, d_ff=4096, n_heads=1, n_layers=1, dropout=0.25).to(device)

    # Load the embeddings
    embeddings_path = cfg.EMBEDDINGS
    input_embeds = []
    for embedding in os.listdir(os.path.join(embeddings_path, activity_name)):
        input_embeds.append(torch.load(os.path.join(embeddings_path, activity_name, embedding))['text'].view(-1))
    input_embeds = torch.stack(input_embeds, dim=0).to(device)
    input_embeds = torch.nn.Embedding.from_pretrained(input_embeds, freeze=True)
    encoder_input = torch.nn.Sequential(input_embeds).to(device)
    batch_size = torch.tensor(np.arange(input_embeds.weight.size(0))).to(device)
    input_embeds = encoder_input(batch_size).to(device)

    if pre_trained is not None:
        net.load_state_dict(torch.load(pre_trained))
        print(f"Pre-trained model loaded from {pre_trained}")

    # Create the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.000001)
 
    # Train the model
    y = train_seq
    net.train()

    for i in range(epochs):

        # warm-up of beta
        if (i + 1) % 100 == 0:
            beta = np.interp(i, [0, epochs], [beta, beta_end])

        net.train()
        optimizer.zero_grad()
        loss1, loss2 = net(input_embeds, y, beta=beta)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            net.eval()
            pred_adjacency_matrix, _ = net.get_adjacency_matrix(input_embeds)
            pred_adjacency_matrix = pred_adjacency_matrix.cpu().numpy()

        # Take the matrix removing first row, first column, last row and las column
        pred_adjacency_matrix = pred_adjacency_matrix[1:-1, 1:-1]
        
        # Threshold the matrix
        pred_adjacency_matrix = np.where(pred_adjacency_matrix < (1/(num_nodes-2)), 0, 1)
        GP = nx.DiGraph(pred_adjacency_matrix)

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

        # Here we use the GT graph to calculate the precision, recall, f1 and accuracy of sequences
        # The GT is never used in the training process
        GT_flatten_adj = nx.adjacency_matrix(GT, nodelist=sorted(GT.nodes)).todense().flatten()
        GP_flatten_adj = nx.adjacency_matrix(GP, nodelist=sorted(GP.nodes)).todense().flatten()
        precision_score = precision(GT_flatten_adj, GP_flatten_adj)
        recall_score = recall(GT_flatten_adj, GP_flatten_adj)
        f1 = f1_score(GT_flatten_adj, GP_flatten_adj)
        pred_predecessors = extract_predecessors(GP)
        accuracy_of_sequences_score = 0
        if nx.is_directed_acyclic_graph(GP):
            accuracy_of_sequences_score = sequences_accuracy(train_seq_no_beta_gamma, pred_predecessors)
        
        print(f"Epoch {i+1}: beta = {beta} | loss1 = {loss1.item()} | loss2 = {loss2.item()} | precision = {precision_score} | recall = {recall_score} | f1 = {f1} | accuracy of sequences = {accuracy_of_sequences_score}")

        if w:
            wandb.log({
                "epoch": i+1,
                "loss1": loss1.item(),
                "loss2": loss2.item(),
                "precision": precision_score,
                "recall": recall_score,
                "f1": f1,
                "accuracy_of_sequences": accuracy_of_sequences_score
            })

    print("Training completed")

    A = nx.nx_agraph.to_agraph(GP)
    A.layout('dot')
    A.draw(os.path.join(output_path, f"{activity_name}", "GP_graph.png"))

    if save:
        torch.save(net.state_dict(), os.path.join(output_path, f"{activity_name}", f"model_{activity_name}-{cfg.SEED}.pt"))
    
    # Test the model
    with torch.no_grad():
        net.eval()
        pred_adjacency_matrix, _ = net.get_adjacency_matrix(input_embeds)
        pred_adjacency_matrix = pred_adjacency_matrix.cpu().numpy()
        pred, _ = net.get_adjacency_matrix(input_embeds)
        pred = pred.cpu().numpy()
        
    # Take the matrix removing first row, first column, last row and las column
    pred_adjacency_matrix = pred_adjacency_matrix[1:-1, 1:-1]
    pred = pred[1:-1, 1:-1]
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
    accuracy_of_sequences_score = 0
    if nx.is_directed_acyclic_graph(GP):
        accuracy_of_sequences_score = sequences_accuracy(train_seq_no_beta_gamma, pred_predecessors)

    # Output the results
    results = {
        "Precision": precision_score,
        "Recall": recall_score,
        "F1": f1,
        "Accuracy of sequences": accuracy_of_sequences_score
    }

    with open(os.path.join(output_path, f"{activity_name}", f"{cfg.SEED}_results.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()