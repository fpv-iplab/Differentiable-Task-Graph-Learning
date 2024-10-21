# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import networkx as nx
import copy
from typing import List, Dict
import numpy as np
import json
import yaml
from box import Box
import glob
import os


def extract_predecessors(G: nx.DiGraph) -> Dict[int, List[int]]:
    """
    Description
    -----------
    Extract the predecessors of the graph.
    
    Parameters
    ----------
    - **G (nx.DiGraph)**: The direct graph.

    Returns
    -------
    - **Dict[int, List[int]]**: The predecessors.
    """
    predecessors = {}
    for current_node in G.nodes:
        predecessors[current_node] = list(G.successors(current_node))
    return predecessors


def delete_redundant_edges(G:nx.DiGraph) -> None:
    """
    Description
    -----------
    Delete the redundant edges of the graph.
    
    Parameters
    ----------
    - **G (nx.DiGraph)**: The direct graph.
    """
    G_copy = copy.deepcopy(G)
    for current_node in G.nodes:
        for anchestor in G_copy.successors(current_node):
            G.remove_edge(current_node, anchestor)
            if not nx.has_path(G, current_node, anchestor):
                G.add_edge(current_node, anchestor)


def sequences_accuracy(sequences:np.ndarray, pred_predecessors:Dict[int, List[int]]) -> float:
    """
    Description
    -----------
    Compute the accuracy of the prediction.

    Parameters
    ----------
    - **sequences (np.ndarray)**: List of sequences.
    - **pred_predecessors (Dict[int, List[int]])**: Dict of predicted predecessors.

    Returns
    -------
    - **float**: The accuracy of sequneces in the current predicted Task Graph.
    """
    accuracy = 0
    for sequence in sequences:
        correct = 0
        for i in range(len(sequence)):
            if len(sequence[:i]) == 0 and len(pred_predecessors[sequence[i]]) == 0:
                correct += 1
            elif len(sequence[:i]) > 0 and len(pred_predecessors[sequence[i]]) > 0:
                # Count the number of correct predecessors
                correct += (len(set(sequence[:i]).intersection(set(pred_predecessors[sequence[i]])))) / len(pred_predecessors[sequence[i]])
        accuracy += correct / len(sequence)
    return accuracy / len(sequences) if len(sequences) > 0 else 0


def load_config_baseline(config_file:str) -> Dict:
    """
    Description
    -----------
    Load config file for baseline.
    
    Parameters
    ----------
    - **config_file (str)**: The config file path.
        
    Returns
    -------
    - **Dict**: A dictionary of the config file.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = Box(config)
    return config.DATA


def load_config_task_graph_learning(config_file:str) -> Dict:
    """
    Description
    -----------
    Load config file for task graph learning.
    
    Parameters
    ----------
    - **config_file (str)**: The config file path.
        
    Returns
    -------
    - **Dict**: A dictionary of the config file.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = Box(config)
    config.TRAIN.ANNOTATIONS = json.load(open(config.TRAIN.ANNOTATIONS))
    config.TRAIN.ACTIVITY_NAME = config.TRAIN.ACTIVITY_NAME.lower().replace(" ", "")
    return config.TRAIN

def load_config_mistake_detection(config_file:str) -> Dict:
    """
    Description
    -----------
    Load config file for mistake detection.
    
    Parameters
    ----------
    - **config_file (str)**: The config file path.
        
    Returns
    -------
    - **Dict**: A dictionary of the config file.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = Box(config)
    config.DATA.ANNOTATIONS = json.load(open(config.DATA.ANNOTATIONS))
    return config.DATA

def load_config_evaluation_baseline(config_file) -> Dict:
    """
    Description
    -----------
    Load config file for evaluation baseline.
    
    Parameters
    ----------
    - **config_file (str)**: The config file path.
        
    Returns
    -------
    - **Dict**: A dictionary of the config file.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = Box(config)
    config.DATA.ANNOTATIONS = json.load(open(config.DATA.ANNOTATIONS))
    config.DATA.GT_TASK_GRAPHS = glob.glob(os.path.join(config.DATA.GT_TASK_GRAPHS, "*.json"))
    config.DATA.PRED_TASK_GRAPHS = glob.glob(os.path.join(config.DATA.PRED_TASK_GRAPHS, "*.adjlist"))
    return config.DATA