# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import numpy as np
import networkx as nx
import torch
import click
import os
import json
import wandb
import random
import copy
import sys

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
@click.option("--log", "-l", type=bool, default=False, is_flag=True, help="Log the output in a file.")
@click.option("--cuda", type=int, default=0, help="CUDA device to use.")
@click.option("-w", type=bool, default=False, is_flag=True, help="Use wandb.")
@click.option("--project_name", type=str, default="TGT", help="Project name for wandb.")
@click.option("--entity", type=str, default=None, help="Entity name for wandb.")
@click.option("--exclude_current", type=bool, default=False, is_flag=True, help="Exclude the current config task graph.")
def main(config:str, log:bool, cuda:str, w:bool, project_name:str, entity:str, exclude_current:bool):
    # Check if wandb is True
    if w:
        if entity is None:
            raise ValueError("You need to specify the entity name.")
        # Initialize wandb
        wandb.login()
        wandb.init(project=project_name, entity=entity)

    # Select the device
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    # Load config
    cfg = load_config_task_graph_learning(config)

    # Activity name (e.g., "coffee")
    activity_name = cfg.ACTIVITY_NAME

    # Epochs
    epochs = cfg.EPOCHS
    if exclude_current:
        epochs = 6000

    # Beta
    beta = 1.0
    beta_end = 0.05

    # Set seed
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    # Set the deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Output path
    output_path = cfg.OUTPUT_DIR
    if exclude_current:
        output_path += "-leave-one-out"

    # Check output path
    if not os.path.exists(os.path.join(output_path, f"{activity_name}")):
        os.makedirs(os.path.join(output_path, f"{activity_name}"))
        
    # Check if log is True
    if log:
        # Redirect the output to the log file
        sys.stdout = open(os.path.join(output_path, f"{activity_name}", "log.txt"), "w")

    # Load all task graphs
    task_graphs = os.listdir(cfg.TASK_GRAPHS)
    task_graphs_json = {}
    for task_graph in task_graphs:
        task_graphs_json[task_graph.split(".")[0]] = json.load(open(os.path.join(cfg.TASK_GRAPHS, task_graph)))
    
    # Annotations
    annotations_json = cfg.ANNOTATIONS["annotations"]

    # Taxonomy
    taxonomy_json = cfg.ANNOTATIONS["taxonomy"]

    # Create NetworkX graphs and other information
    GTs = {}

    for task_graph in task_graphs_json:
        # Exclude the current task graph
        # This is used to train the model with the other task graphs
        if exclude_current and task_graph == activity_name:
            continue
        
        # Ground truth graph
        GT = nx.DiGraph()
        for node in task_graphs_json[task_graph]["steps"]:
            GT.add_node(int(node), label=task_graphs_json[task_graph]["steps"][node], shape="box")
        for edge in task_graphs_json[task_graph]["edges"]:
            GT.add_edge(edge[1], edge[0])

        # ID beta
        id_beta = 0

        # ID START
        id_start = 1

        # ID END
        id_end = len(GT.nodes)

        # ID gamma
        id_gamma = len(GT.nodes) + 1

        # Number of nodes
        num_nodes = len(GT.nodes) + 2

        # Activity name (e.g., "coffee")
        activity_name_graph = task_graph

        # Get the mapping from the description to the id of the keysteps inside the taxonomy
        description_to_id = {}
        for step_id in taxonomy_json[activity_name_graph]:
            description_to_id[taxonomy_json[activity_name_graph][step_id]["name"]] = int(step_id)
            
        # Get the mapping from the node to the id
        mapping_nodes = {}
        for node in GT.nodes:
            if GT.nodes[node]["label"] in description_to_id:
                mapping_nodes[node] = int(description_to_id[GT.nodes[node]["label"]])

        # Get the mapping from the id to the node
        id_to_node = {mapping_nodes[node]: node for node in mapping_nodes}

        # Get train sequences
        train_sequences = []
        for video in annotations_json:
            if annotations_json[video]["scenario"] != activity_name_graph:
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

        # Input embeds
        embeddings_path = cfg.EMBEDDINGS
        input_embeds = []
        for embedding in os.listdir(os.path.join(embeddings_path, activity_name_graph)):
            input_embeds.append(torch.load(os.path.join(embeddings_path, activity_name_graph, embedding))['text'].view(-1))
        input_embeds_ = torch.stack(input_embeds, dim=0)
        input_embeds = torch.nn.Embedding.from_pretrained(input_embeds_, freeze=True)
        encoder_input = torch.nn.Sequential(input_embeds)
        batch_size = torch.tensor(np.arange(input_embeds.weight.size(0)))

        GTs[task_graph] = {
            "GT": GT,
            "num_nodes": num_nodes,
            "id_start": id_start,
            "id_end": id_end,
            "id_beta": id_beta,
            "id_gamma": id_gamma,
            "encoder_input": encoder_input,
            "batch_size": batch_size,
            "train_sequences": train_seq,
            "train_seq_no_beta_gamma": train_seq_no_beta_gamma
        }
    
    print("Task graphs loaded")
    print("Number of task graphs: ", len(GTs))

    # Create the model
    net = TGT(device=device, dropout=0.25).to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.000001)
 
    # Train the model
    net.train()

    for i in range(epochs):

        # warm-up of the beta parameter
        if (i + 1) % 100 == 0:
            beta = np.interp(i, [0, epochs], [beta, beta_end])

        net.train()

        precision_mean = 0
        recall_mean = 0
        f1_mean = 0
        accuracy_of_sequences_score_mean = 0
        best_accuracy_of_sequences_score = -float("inf")
        best_model = None
        loss1_mean = 0
        loss2_mean = 0
        loss_mean = 0

        for graph in GTs:
            GT = GTs[graph]["GT"]
            num_nodes = GTs[graph]["num_nodes"]
            id_start = GTs[graph]["id_start"]
            id_end = GTs[graph]["id_end"]
            encoder_input = GTs[graph]["encoder_input"]
            batch_size = GTs[graph]["batch_size"]
            y = GTs[graph]["train_sequences"]
            train_seq_no_beta_gamma = GTs[graph]["train_seq_no_beta_gamma"]

            input_embeds = encoder_input(batch_size).to(device)
            loss1, loss2 = net(input_embeds, y, beta=beta)
            
            # Sum the losses for all the graphs
            loss_mean += (loss1 + loss2)

            with torch.no_grad():
                net.eval()
                pred_adjacency_matrix, _ = net.get_adjacency_matrix(input_embeds)
                pred_adjacency_matrix = pred_adjacency_matrix.cpu().numpy()

            # Take the matrix removing first row, first column, last row and las column
            pred_adjacency_matrix = pred_adjacency_matrix[1:-1, 1:-1]
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

            precision_mean += precision_score
            recall_mean += recall_score
            f1_mean += f1
            accuracy_of_sequences_score_mean += accuracy_of_sequences_score
            loss1_mean += loss1.item()
            loss2_mean += loss2.item()

        precision_mean /= (len(GTs))
        recall_mean /= (len(GTs))
        f1_mean /= (len(GTs))
        accuracy_of_sequences_score_mean /= (len(GTs))
        # Sum the losses for all the graphs and divide by the number of graphs
        loss1_mean /= (len(GTs))
        loss2_mean /= (len(GTs))
        loss_mean /= (len(GTs))
        
        # Backward
        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

        if accuracy_of_sequences_score_mean >= best_accuracy_of_sequences_score:
            best_accuracy_of_sequences_score = accuracy_of_sequences_score_mean
            best_model = copy.deepcopy(net.state_dict())

        print(f"Epoch {i+1}: beta = {beta} | loss1 = {loss1_mean} | loss2 = {loss2_mean} | precision = {precision_mean} | recall = {recall_mean} | f1 = {f1_mean} | accuracy of sequences = {accuracy_of_sequences_score_mean}")

        if w:
            wandb.log({"epoch": i+1, "loss1_mean": loss1_mean, "loss2_mean": loss2_mean, "precision_mean": precision_mean, "recall_mean": recall_mean, "f1_mean": f1_mean, "accuracy_of_sequences_mean": accuracy_of_sequences_score_mean})
    
    print("Training completed")

    if best_model is not None:
        net.load_state_dict(best_model)

    # Save the model
    torch.save(net.state_dict(), os.path.join(output_path, f"{activity_name}", f"model_unified.pth"))

if __name__ == "__main__":
    main()