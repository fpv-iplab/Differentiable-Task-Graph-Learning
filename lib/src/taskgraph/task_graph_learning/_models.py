import torch
import torch.nn as nn
import numpy as np

from ._loss import task_graph_maximum_likelihood_loss
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TaskGraph(nn.Module):
    """Task Graph model"""
    def __init__(self, num_nodes):
        super(TaskGraph, self).__init__()
        self.task_graph = nn.Parameter(torch.rand(num_nodes, num_nodes))

    def forward(self):
        return self.task_graph
    

class DO(nn.Module):
    """Direct Optimization model"""
    def __init__(self, num_nodes, device='cpu'):
        super(DO, self).__init__()
        self.adjacency_matrix = TaskGraph(num_nodes)
        self.softmax = nn.Softmax(dim=1)
        self.size = num_nodes
        self.device = device

    def get_adjacency_matrix(self):
        mask = np.ones((self.size, self.size), dtype=bool)
        mask[0, :] = False
        mask[1, :] = False
        mask[1, 0] = True
        mask[:, -1] = False
        mask[:, -2] = False
        mask[-1, :] = False
        mask[-1, -2] = True
        action_mask = (mask) & (~np.identity(self.size, dtype=bool))
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        values = self.adjacency_matrix()
        softmax_input = torch.where(action_mask, values, torch.tensor(-float('inf')).to(self.device))
        values = self.softmax(softmax_input)
        return values
        
    def forward(self, y, eps=1e-6, beta=0.005):
        values = self.get_adjacency_matrix()
        # We sum eps to avoid log(0)
        return task_graph_maximum_likelihood_loss(y, values + eps, beta)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super(Encoder, self).__init__()
        encoder_layers = TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
    
    def forward(self, src):
        output = self.transformer_encoder(src)
        return output

class TGT(nn.Module):
    def __init__(self, d_model=4096, n_heads=1, n_layers=1, d_ff=4096, dropout=0.1, device='cpu'):
        super(TGT, self).__init__()
        self.embedding_start = nn.Embedding(2, d_model)
        self.embedding_end = nn.Embedding(2, d_model)
        self.transformer = Encoder(d_model, d_ff, n_layers, n_heads, dropout)
        self.f = nn.Sequential(
            nn.TransformerEncoderLayer(d_model*2, nhead=2, dropout=dropout, dim_feedforward=2048),
            nn.Linear(d_model*2, 16),
            nn.TransformerEncoderLayer(16, nhead=2, dropout=dropout, dim_feedforward=2048),
            nn.Linear(16, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.device = device
    
    def get_adjacency_matrix(self, vectors, custom_mask=None):
        vectors = torch.cat((self.embedding_start.weight, vectors, self.embedding_end.weight), dim=0)
        vectors = self.transformer(vectors.unsqueeze(0)).squeeze()
        vectors = vectors.to(self.device)
        ii,jj=torch.meshgrid(torch.arange(len(vectors)),torch.arange(len(vectors)), indexing='ij')
        ii = ii.ravel()
        jj = jj.ravel()
        rel_features=torch.cat([vectors[ii],vectors[jj]],-1)
        out = self.f(rel_features)
        values = out.view(len(vectors),len(vectors))
        mask = np.ones((values.size()[0], values.size()[0]), dtype=bool)
        mask[0, :] = False
        mask[1, :] = False
        mask[1, 0] = True
        mask[:, -1] = False
        mask[:, -2] = False
        mask[-1, :] = False
        mask[-1, -2] = True
        if custom_mask is not None:
            action_mask = (mask) & (~np.identity(values.size()[0], dtype=bool)) & (custom_mask)
        else:
            action_mask = (mask) & (~np.identity(values.size()[0], dtype=bool))
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        large_negative = torch.tensor(-1e10).to(self.device)
        softmax_input = torch.where(action_mask, values, large_negative)
        values = self.softmax(softmax_input)
        return values, vectors

    def forward(self, vectors, y, eps, beta, mask=None):
        values, vectors = self.get_adjacency_matrix(vectors, mask)
        vectors = F.normalize(vectors, p=2, dim=-1)
        vectors = vectors @ vectors.T
        vectors *= torch.exp(torch.tensor(0.9).to(self.device))
        labels = torch.arange(len(vectors), device = self.device) 
        loss = F.cross_entropy(vectors, labels) * (len(vectors))
        return task_graph_maximum_likelihood_loss(y, values + eps, beta), loss