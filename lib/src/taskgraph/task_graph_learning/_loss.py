# This is the implementation of our defined Task Graph Maximum Likelihood Loss function.

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

def task_graph_rate(sequence, A, all_nodes, beta):
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

    with ThreadPoolExecutor(max_workers=len(sequence) - 1) as executor:
        futures = [executor.submit(compute_rate, i, sequence_np, A, all_nodes) for i in range(1, len(sequence)-1)]
        for future in futures:
            s += future.result()

    return s

def task_graph_maximum_likelihood_loss(y, A, beta):
    def compute_rate(s, all_nodes):
        return task_graph_rate(s, A, all_nodes, beta) 
    
    losses = []
    all_nodes = list(range(A.shape[0]))
    all_nodes = np.array(all_nodes[1:-1])
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(compute_rate, s, all_nodes) for s in y]
        for future in futures:
            rate = future.result()
            losses.append(rate)
            
    loss = torch.sum(torch.stack(losses))
    return loss