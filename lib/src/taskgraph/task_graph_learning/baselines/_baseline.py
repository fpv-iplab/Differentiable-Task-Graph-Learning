# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

# It contains the implementation of the baselines used in the paper.

import os
import copy
import networkx as nx
import pygraphviz as pgv
import numpy as np
from collections import defaultdict
from ._transition_graph import TransitionGraph

def get_preconditions(scenario:str, actions_names:dict[str, list], data:dict[str, dict], baseline_func:callable):
    """
    Description
    -----------
    Get the preconditions of the actions of the given scenario.

    Parameters
    ----------
    - **scenario (str)**: The name of the scenario.
    - **actions_names (dict[str, list])**: A dictionary containing the actions and their names.
    - **data (dict[str, dict])**: The annotations of the videos.
    - **baseline_func (callable)**: The baseline function.

    Returns
    ------- 
    The preconditions of the actions of the given scenario.
    """
    # Actions id in the scenario
    actions = actions_names["actions"]

    # Actions in the videos of the scenario
    actions_videos = []
    for video in data:
        if data[video]["scenario"] != scenario:
            continue
        sequence = []
        for segment in data[video]["segments"]:
            if segment["step_id"] not in sequence:
                sequence.append(segment["step_id"])
        actions_videos.append(sequence)
    
    return baseline_func(actions, actions_videos)



def create_eligibility_vectors(sequences:list) -> dict:
    """
    Description
    -----------
    Create the eligibility vectors of the actions of the given scenario.
    
    Parameters
    ----------
    - **sequences (List)**: The actions in the videos of the scenario.
        
    Returns
    -------
    - **dict**: The eligibility vectors of the actions of the given scenario.
    """
    # Initialize eligibility vectors dictionary
    eligibility_vectors = defaultdict(list)
    nodes = sorted(set(node for seq in sequences for node in seq))
    for sequence in sequences:
        for i, node in enumerate(sequence):
            ev = np.zeros(len(nodes))
            for preceding_node in sequence[:i]:
                index = nodes.index(preceding_node)
                ev[index] = 1
            eligibility_vectors[node].append(ev)
    return eligibility_vectors



def iteration_ILP(sequences:list[list[int]], eligibility_vectors_dict:dict, objective:str="acc", alpha:int=0) -> tuple[dict, dict]:
    """
    Description
    -----------
    The code implements:
    - Eq. 6 of [1], credited as (Muggleton, 1991) and (Sohn et al., 2020):
        this is obtained by specifying objective='acc'.
    - Eq. 7 of [1]: this is obtained by specifying objective='prec' and alpha=0.
    - Eq. 8 of [1]: this is obtained by specifying objective='prec' and alpha>0.
    
    References
    ----------
    - [1] Jang, Yunseok, et al. "Multimodal subtask graph generation from instructional videos." arXiv preprint arXiv:2302.08672 (2023).

    Parameters
    ----------
    - **sequences (List[List[int]])**: The actions in the videos of the scenario.
    - **eligibility_vectors_dict (dict)**: The eligibility vectors of the actions of the given scenario.
    - **objective (str)**: *(Optional)* The objective function. Defaults to "acc".
    - **alpha (int)**: *(Optional)* The alpha value. Defaults to 0.

    Returns
    -------
    - **tuple[dict, dict]**: The preconditions of the actions of the given scenario.
    """
    preconditions = defaultdict(list)
    precondition_scores = defaultdict(list)
    nodes = sorted(eligibility_vectors_dict.keys())

    # Cycle through eligibility vectors e, our training data
    for node, evs in eligibility_vectors_dict.items():
        # Dependencies found for this node and related scores
        # These allows us to compute function f
        dependencies = np.zeros(len(eligibility_vectors_dict)) #initially zero
        scores = np.zeros(len(eligibility_vectors_dict)) #initially zero

        # iterative search
        while True:
            best_node = None # the current best node to add to the list of dependencies is unknown
            max_score = -1 # the score of this choice is zero

            for nd in nodes: # cycle through all nodes and check what's the advantage of adding this single node
                tmp_deps = dependencies.copy() # clone current list of dependencies
                if tmp_deps[nodes.index(nd)]: # if the current node is already in, let's skip this iteration
                    continue

                # Add current node to the list of dependencies
                tmp_deps[nodes.index(nd)] = 1

                # tmp_deps now defines a function f which can be used to determine whether
                # the current node "node" can be executed given a set of completed steps
                # in practice, we use this function to compute a score of how good
                # our predictions are, by comparing them to our training data, the eligibility vectors "evs"
                # crucially, we only have positive examples in our set
                if objective == 'acc':
                    # compute the accuracy as the fraction of eligibility vectors
                    # in which all predicted preconditions are satisfied
                    y_pred = [tmp_deps for _ in range(len(evs))]
                    score = np.logical_and(np.concatenate(np.array(evs)), np.concatenate(np.array(y_pred))).sum()
                elif objective == 'prec':
                    y_pred = [tmp_deps for _ in range(len(evs))]
                    num = np.logical_and(np.concatenate(np.array(evs)), np.concatenate(np.array(y_pred))).sum()
                    den = np.sum(np.concatenate(np.array(y_pred)))
                    if num > 0 and den > 0:
                        score = num/den
                    else:
                        score = 0

                if alpha > 0:
                    score -= alpha*np.sum(tmp_deps)

                if max_score < score:
                    max_score = score
                    best_node = nodes.index(nd)

            if max_score != -1:
                dependencies[best_node] = 1 # permanent
                scores[best_node] = max_score
            else:
                break

        preconditions[node] = [nodes[i] for i in dependencies.nonzero()[0]]
        precondition_scores[node] = [scores[i] for i in dependencies.nonzero()[0]]

    return preconditions, precondition_scores



def delete_simple_cycles(G:nx.DiGraph, precondition_scores:dict, preconditions:dict) -> bool:
    """
    Description
    -----------
    Check if the graph contains simple cycles.
    
    Parameters
    ----------
    - **G (nx.DiGraph)**: The graph.
    - **precondition_scores (dict)**: The scores of the preconditions.
    - **preconditions (dict)**: The preconditions.
    
    Returns
    -------
    - **bool**: True if the graph is a DAG after removing the simple cycles, False otherwise.
    """
    G_copy = copy.deepcopy(G)
    already_removed = []
    if len(G_copy.edges) > len(G_copy.nodes) * (len(G_copy.nodes) - 1) / 2:
        return False
    for cycle in nx.simple_cycles(G_copy):
        if len(cycle) > 2:
            continue
        if cycle in already_removed:
            continue
        already_removed.append(cycle)
        id_1 = cycle[0]
        id_2 = cycle[1]
        if precondition_scores[id_1][preconditions[id_1].index(id_2)] >= precondition_scores[id_2][preconditions[id_2].index(id_1)]:
            G.remove_edge(id_2, id_1)
        else:
            G.remove_edge(id_1, id_2)
    return nx.is_directed_acyclic_graph(G)



def baseline_ILP(sequences:list[list[int]], objective:str="acc", thresh:int=0) -> tuple[dict, dict, nx.DiGraph, dict, dict]:
    """
    Description
    -----------
    Calculate the preconditions of the actions of the given scenario using the ILP baseline.
    
    Parameters
    ----------
    - **sequences (list[list[int]])**: The actions in the videos of the scenario.
    - **objective (str)**: *(Optional)* The objective function. Defaults to "acc".
    - **thresh (int)**: *(Optional)* The threshold. Defaults to 0.

    Returns
    -------
    - **tuple[dict, dict, nx.DiGraph, dict, dict]**: The preconditions of the actions of the given scenario.
    """
    alpha = 0
    eligibility_vectors_dict = create_eligibility_vectors(sequences)
    preconditions, precondition_scores = iteration_ILP(sequences, eligibility_vectors_dict, objective, alpha)
    while True:
        graph = nx.DiGraph()
        for v in preconditions.keys():
            graph.add_node(v)
        for k,vv in preconditions.items():
            if len(precondition_scores[k]) == 0:
                continue
            max_score = max(precondition_scores[k])
            if max_score < thresh:
                continue
            for v, s in zip(vv, precondition_scores[k]):
                if s == max_score and v != k:
                    graph.add_edge(k, v)
        if delete_simple_cycles(graph, precondition_scores, preconditions):
            # Delete redundant edges
            delete_redundant_edges(graph)
            # Take all sink nodes
            sink_nodes = [node for node in graph.nodes if graph.in_degree(node) == 0]
            # Take all source nodes
            source_nodes = [node for node in graph.nodes if graph.out_degree(node) == 0]
            # Add START and END nodes
            graph.add_node("START")
            graph.add_node("END")

            for node in source_nodes:
                graph.add_edge(node, "START")
            for node in sink_nodes:
                graph.add_edge("END", node)
            break
        else:
            alpha += 0.001
            preconditions, precondition_scores = iteration_ILP(sequences, eligibility_vectors_dict, objective, alpha)
    mapping_node_to_id = {}
    mapping_node_to_id["START"] = 0
    mapping_node_to_id["END"] = len(graph.nodes) - 1
    count = 1
    for node in graph.nodes:
        if node not in mapping_node_to_id:
            mapping_node_to_id[node] = count
            count += 1
    mapping_id_to_node = {v: k for k, v in mapping_node_to_id.items()}
    return preconditions, precondition_scores, graph, mapping_node_to_id, mapping_id_to_node



def baseline_transition_graph(keystep_json:str, scenario:str, max_length:int) -> TransitionGraph:
    """
    Description
    -----------
    Create the transition graph of the preconditions of the actions of the given scenario.
    
    Parameters
    ----------
    - **keystep_json (str)**: The path to the keystep json file.
    - **scenario (str)**: The name of the scenario.
    - **max_length (int)**: The max length of the sequences to be considered.

    Returns
    -------
    - **TransitionGraph**: The transition graph of the preconditions of the actions of the given scenario.
    """
    return TransitionGraph(keystep_json, scenario, max_length)



def create_graph(preconditions:dict[str, list[int]], scenario:str, actions:list[int], names:list[str], save:bool=False, output:str="./") -> nx.DiGraph:
    """
    Description
    -----------
    Create the graph of the preconditions of the actions of the given scenario.

    Parameters
    ----------
    - **preconditions (dict[str, list[int]])**: The preconditions of the actions of the given scenario.
    - **scenario (str)**: The name of the scenario.
    - **actions (list[int])**: The actions id in the scenario.
    - **names (list[str])**: The actions names in the scenario.
    - **save (bool)**: *(Optional)* If True save the graph as svg. Defaults to False.
    - **output (str)**: *(Optional)* The output directory. Defaults to "./".

    Returns
    -------
    - **nx.DiGraph**: The graph of the preconditions of the actions of the given scenario.
    """
    # Create the graph
    G = nx.DiGraph()
    for name in names:
        node_name = str(actions[names.index(name)]) + "_" + name
        G.add_node(node_name, info=name, node_color='lightblue', node_shape='')
    for action in preconditions:
        for precondition in preconditions[action]:
            if precondition == "START":
                continue
            src = str(action) + "_" + names[actions.index(action)]
            dst = str(precondition) + "_" + names[actions.index(precondition)]
            G.add_edge(src, dst)
    delete_redundant_edges(G)
    delete_cycles(G)
    insert_start_end_nodes(G)
    if save:
        save_graph_as_svg(G, scenario, output)
    return G



def delete_cycles(G:nx.DiGraph) -> None:
    """
    Description
    -----------
    Delete the cycles of the graph.
    
    Parameters
    ----------
    - **G (nx.DiGraph)**: The direct graph.
    """
    while not nx.is_directed_acyclic_graph(G):
        # Find a cycles in a directed graph
        cycle = nx.find_cycle(G, orientation='original')
        # Remove the cycle
        G.remove_edge(cycle[-1][0], cycle[-1][1])



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



def insert_start_end_nodes(G:nx.DiGraph) -> None:
    """
    Description
    -----------
    Insert the start and end nodes in the graph.
    
    Parameters
    ----------
    - **G (nx.DiGraph)**: The direct graph.
    """
    # Insert the END node
    G.add_node("END", info="END", node_color='lightblue', node_shape='')
    for node in G.nodes:
        if G.in_degree(node) == 0 and node != "END":
            G.add_edge("END", node)
    # Insert the START node
    G.add_node("START", info="START", node_color='lightblue', node_shape='')
    for node in G.nodes:
        if G.out_degree(node) == 0 and node != "START":
            G.add_edge(node, "START")                    



def save_graph_as_svg(G:nx.DiGraph, scenario:str, output:str) -> None:
    """
    Description
    -----------
    Save the graph as svg.

    Parameters
    ----------
    - **G (nx.DiGraph)**: The direct graph.
    - **scenario (str)**: The name of the scenario.
    - **output (str)**: The output directory.
    """
    # Set the shape of the nodes
    for node in G.nodes:
        G.nodes[node]["shape"] = "box"

    # Use PyGraphviz to render the graph as an SVG file
    G_pg = nx.nx_agraph.to_agraph(G)
    G_pg.layout(prog='dot', args='-Grankdir=UD')

    # Save the graph as a DOT file
    name = scenario.replace(" ", "_").lower()
    name = name.replace("&", "and")
    name = "graph_" + name
    path_dot = os.path.join(output, name + '.dot')
    G_pg.write(path_dot)

    # Save the networkx graph
    path_adjlist = os.path.join(output, name + '.adjlist')
    nx.write_multiline_adjlist(G, path_adjlist, delimiter=';')

    # Use PyGraphviz to render the graph as an PNG file
    G_pg = pgv.AGraph(path_dot)
    path_svg = os.path.join(output, name + '.png')
    G_pg.draw(path_svg, prog='dot', format='png')