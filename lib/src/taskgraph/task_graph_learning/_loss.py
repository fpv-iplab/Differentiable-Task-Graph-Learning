# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

# This is the implementation of our defined Task Graph Maximum Likelihood Loss function.

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

def task_graph_rate(sequence, A, all_nodes, beta):
    """
    Description
    -----------
    Compute the rate of a sequence of nodes in a task graph.
    
    Parameters
    ----------
    - **sequence (np.ndarray)**: The sequence of nodes.
    - **A (np.ndarray)**: The adjacency matrix of the task graph.
    - **all_nodes (np.ndarray)**: The list of all nodes in the task graph.
    - **beta (float)**: The beta parameter.
        
    Returns
    -------
    - **float**: The rate of the sequence.
    """
    def compute_rate(i, sequence_np, A, all_nodes):
        current = sequence_np[i]
        mask = np.ones(len(all_nodes), dtype=bool)
        mask[np.isin(all_nodes, sequence_np[:i])] = False
        J = sequence_np[:i]
        notJ = all_nodes[mask]
        num = A[current, J].sum()
        den = A[notJ, :][:, J].sum()
        return beta * torch.log(den) - torch.log(num)
    s = 0
    if type(sequence) == torch.Tensor:
        sequence_np = sequence.cpu().numpy() 
    else:
        sequence_np = sequence

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(compute_rate, i, sequence_np, A, all_nodes) for i in range(1, len(sequence)-1)]
        for future in futures:
            s += future.result()

    return s

def task_graph_maximum_likelihood_loss(y, A, beta):
    """
    Description
    -----------
    Compute the maximum likelihood loss of a sequence of nodes in a task graph.
    
    Parameters
    ----------
    - **y (np.ndarray)**: The sequence of nodes.
    - **A (np.ndarray)**: The adjacency matrix of the task graph.
    - **beta (float)**: The beta parameter.
        
    Returns
    -------
    - **torch.Tensor**: The loss of the sequence.
    """
    def compute_rate(s, all_nodes):
        # Here we delete the repeated steps in the sequence with a probability of 0.5
        # This allows us to have a more robust model.
        seq = []
        for keystep in s:
            if keystep not in seq:
                seq.append(keystep)
            elif np.random.uniform() < 0.5:
                seq.remove(keystep)
                seq.append(keystep)
        return task_graph_rate(seq, A, all_nodes, beta) 
    
    losses = []
    all_nodes = list(range(A.shape[0]))
    all_nodes = np.array(all_nodes[1:-1])
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(compute_rate, s, all_nodes) for s in y]
        for future in futures:
            rate = future.result()
            losses.append(rate)
            
    loss = torch.sum(torch.stack(losses))
    return loss