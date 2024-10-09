# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import json
import copy
import pandas as pd
import networkx as nx

class TransitionGraph:
    """
    Description
    -----------
    TransitionGraph class to represent the transition graph of a scenario.
    """

    def __init__(self, keystep_json:str, scenario:str, max_length:int=-1) -> None:
        """
        Description
        -----------
        Init function for the TransitionGraph class.

        Parameters
        ----------
        - **keystep_json (str)**: The path to the keystep json file.
        - **scenario (str)**: The scenario to be analyzed.
        - **max_length (int)**: *(Optional)* The maximum length of the sequences to be considered. Default is -1.
        """
        self.keystep_json = keystep_json
        self.scenario = scenario
        # Check if the keystep json is a csv file
        if not self.keystep_json.endswith('.csv'):
            self.setup(max_length)
        else:
            transition_matrix_ = pd.read_csv(self.keystep_json, index_col=0)
            self.actions = transition_matrix_.index.tolist()
            for i in range(len(self.actions)):
                try:
                    self.actions[i] = int(self.actions[i])
                except:
                    continue
            self.transition_matrix = pd.DataFrame(0, index=self.actions, columns=self.actions)
            for row in transition_matrix_.index:
                for col in transition_matrix_.columns:
                    if (row == "START" or row == "END") and (col == "START" or col == "END"):
                        self.transition_matrix.at[row, col] = transition_matrix_.at[row, col]
                    elif row == "START" or row == "END":
                        self.transition_matrix.at[row, int(col)] = transition_matrix_.at[row, col]
                    elif col == "START" or col == "END":
                        self.transition_matrix.at[int(row), col] = transition_matrix_.at[row, col]
                    else:
                        self.transition_matrix.at[int(row), int(col)] = transition_matrix_.at[row, col]
    
    def setup(self, max_length:int) -> None:
        """
        Description
        -----------
        Setup function for TransitionGraph class. 
        This function will find the scenario in the keystep json file, 
        create a new json file with only the scenario, and parse the mermaid file.
        
        Parameters
        ----------
        - **max_length (int)**: The maximum length of the sequences to be considered.
        """
        self.find_scenario(max_length)
        self.create_id_to_step()
        self.create_step_to_id()
        self.create_transition_matrix()

    def find_scenario(self, max_length:int) -> None:
        """
        Description
        -----------
        Find the scenario in the keystep json file and create a new json file with only the scenario.
        
        Parameters
        ----------
        - **max_length (int)**: The maximum length of the sequences to be considered.
        """
        self.dict_scenario = json.load(open(self.keystep_json))
        dict_scenario_ = copy.deepcopy(self.dict_scenario)
        for key in dict_scenario_["annotations"].keys():
            if self.dict_scenario["annotations"][key]["scenario"] != self.scenario:
                del self.dict_scenario["annotations"][key]
        if max_length != -1:
            if len(self.dict_scenario["annotations"]) > max_length:
                self.dict_scenario["annotations"] = dict(list(self.dict_scenario["annotations"].items())[:max_length])

    def create_id_to_step(self) -> None:
        """
        Description
        -----------
        Create a dictionary that maps the id of a step to the step name.
        """
        self.id_to_step = {}
        for d in self.dict_scenario["taxonomy"][self.scenario]:
            d = self.dict_scenario["taxonomy"][self.scenario][d]
            self.id_to_step[d["id"]] = str(d["id"]) + "_" + d["name"] 

    def create_step_to_id(self) -> None:
        """
        Description
        -----------
        Create a dictionary that maps the step name to the id of the step.
        """
        self.step_to_id = {v : k for k, v in self.id_to_step.items()}
        
    def create_transition_matrix(self) -> None:
        """
        Description
        -----------
        Create a transition matrix from the actions in the scenario.
        """
        self.actions = []
        for key in self.dict_scenario["annotations"].keys():
            for segment in self.dict_scenario["annotations"][key]["segments"]:
                if segment["step_id"] in self.id_to_step.keys() and segment["step_id"] not in self.actions:
                    self.actions.append(segment["step_id"])
        actions = ["START"]
        actions.extend(self.actions)
        actions.append("END")
        self.actions = actions
        self.transition_matrix = pd.DataFrame(0, index=self.actions, columns=self.actions)
        for key in self.dict_scenario["annotations"].keys():
            steps = []
            steps.append("START")
            for segment in self.dict_scenario["annotations"][key]["segments"]:
                if segment["step_id"] in self.id_to_step.keys():
                    if self.id_to_step[segment["step_id"]] not in steps:
                        steps.append(segment["step_id"])
            for i in range(len(steps)-1):
                self.transition_matrix.loc[steps[i], steps[i+1]] += 1
            self.transition_matrix.loc[steps[-1], "END"] += 1
        self.transition_matrix = self.transition_matrix.div(self.transition_matrix.sum(axis=1), axis=0)
        self.transition_matrix = self.transition_matrix.fillna(0)

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Description
        -----------
        Get the transition matrix of the scenario.

        Returns
        -------
        - **pd.DataFrame**: The transition matrix of the scenario.
        """
        return self.transition_matrix
    
    def save_transition_matrix(self, output_filename:str) -> None:
        """
        Description
        -----------
        Save the transition matrix to a csv file.

        Parameters
        ----------
        - **output_filename (str)**: The name of the output file.
        """
        self.transition_matrix.to_csv(output_filename)
    
    def get_preconditions_scores(self, step_id:int) -> dict:
        """
        Description
        -----------
        Get the past steps of a given step id.

        Parameters
        ----------
        - **step_id (int)**: The step id.

        Returns
        -------
        - **dict**: A dictionary with the past steps and their conditional probability.
        """
        past_steps = {}
        if step_id not in self.actions:
            return past_steps
        for action_A in self.actions:
            # Calculate the conditional probability P(pred=action_A | succ=action_X)
            conditional_prob_A_given_X = 0.0  # Initialize the conditional probability to a default value
            # Check if the sum of transition probabilities for the current step_id is not zero
            if self.transition_matrix[step_id].sum() != 0:
                conditional_prob_A_given_X = self.transition_matrix.at[action_A, step_id] / self.transition_matrix[step_id].sum()
            if conditional_prob_A_given_X > 0:
                past_steps[action_A] = conditional_prob_A_given_X
        # Sort the dictionary by the conditional probability
        past_steps = {k: v for k, v in sorted(past_steps.items(), key=lambda item: item[1], reverse=True)}
        return past_steps
    
    def get_future_scores(self, step_id:int) -> dict:
        """
        Description
        -----------
        Get the future steps of a given step id.

        Parameters
        ----------
        - **step_id (int)**: The step id.

        Returns
        -------
        - **dict**: A dictionary with the future steps and their conditional probability.
        """
        future_steps = {}
        if step_id not in self.actions:
            return future_steps
        for action_A in self.actions:
            # Calculate the conditional probability P(succ=action_A | pred=action_X)
            conditional_prob_A_given_X = self.transition_matrix.at[step_id, action_A]
            if conditional_prob_A_given_X > 0:
                future_steps[action_A] = conditional_prob_A_given_X
        # Sort the dictionary by the conditional probability
        future_steps = {k: v for k, v in sorted(future_steps.items(), key=lambda item: item[1], reverse=True)}
        return future_steps

    def create_networkx_graph(self) -> tuple[nx.DiGraph, dict]:
        """
        Description
        -----------
        Create a networkx graph from the transition matrix.

        Returns
        -------
        - **tuple(nx.DiGraph, dict)**: A networkx graph and a dictionary with the edge labels.
        """
        G = nx.DiGraph()
        edge_labels = {}
        for action in self.actions:
            G.add_node(self.id_to_step[action] if (action != "START" and action != "END") else action)
            preconditions = self.get_preconditions_scores(action)
            max_score = max(preconditions.values()) if len(preconditions) > 0 else 0
            for precondition in preconditions:
                if preconditions[precondition] < max_score:
                    continue
                r = self.id_to_step[precondition] if (precondition != "START" and precondition != "END") else precondition
                c = self.id_to_step[action] if (action != "START" and action != "END") else action
                edge_labels[(r, c)] = f"{preconditions[precondition]*100:.2f}%"
                G.add_edge(r, c, label=preconditions[precondition]*100)
        
        # Remove cycles from the graph with minimum probability
        for cycle in nx.simple_cycles(G):
            if self.step_to_id[cycle[0]] == self.step_to_id[cycle[-1]]:
                G.remove_edge(cycle[0], cycle[-1])
                continue
            min_prob = 1
            min_edge = None
            for i in range(len(cycle)-1):
                if self.transition_matrix.at[self.step_to_id[cycle[i]], self.step_to_id[cycle[i+1]]] < min_prob:
                    min_prob = self.transition_matrix.at[self.step_to_id[cycle[i]], self.step_to_id[cycle[i+1]]]
                    min_edge = (cycle[i], cycle[i+1])
            if min_edge is not None and G.has_edge(min_edge[0], min_edge[1]):
                G.remove_edge(min_edge[0], min_edge[1])

        # If there aren't any ingoing node check the best probability with no cycles
        for node in G.nodes():
            if len(list(G.predecessors(node))) == 0 and node != "START":
                preconditions = self.get_preconditions_scores(self.step_to_id[node])
                for precondition in preconditions:
                    r = self.id_to_step[precondition] if (precondition != "START" and precondition != "END") else precondition
                    c = self.id_to_step[self.step_to_id[node]] if (self.step_to_id[node] != "START" and self.step_to_id[node] != "END") else self.step_to_id[node]
                    edge_labels[(r, c)] = f"{preconditions[precondition]*100:.2f}%"
                    G.add_edge(r, c, label=preconditions[precondition]*100)
                    if not nx.is_directed_acyclic_graph(G):
                        G.remove_edge(r, c)
                    else:
                        break

        # If there aren't any outgoing node check the best probability with no cycles
        for node in G.nodes():
            if len(list(G.successors(node))) == 0 and node != "END":
                future_scores = self.get_future_scores(self.step_to_id[node])
                for future_score in future_scores:
                    r = self.id_to_step[self.step_to_id[node]] if (self.step_to_id[node] != "START" and self.step_to_id[node] != "END") else self.step_to_id[node]
                    c = self.id_to_step[future_score] if (future_score != "START" and future_score != "END") else future_score
                    edge_labels[(r, c)] = f"{future_scores[future_score]*100:.2f}%"
                    G.add_edge(r, c, label=future_scores[future_score]*100)
                    if not nx.is_directed_acyclic_graph(G):
                        G.remove_edge(r, c)
                    else:
                        break

        in_degree_zeros = [node for node in G.nodes if G.in_degree(node) == 0 and node != "START" and node != "END"]
        for node in in_degree_zeros:
            G.add_edge("START", node, label=100)
               
        out_degree_zeros = [node for node in G.nodes if G.out_degree(node) == 0 and node != "START" and node != "END"]
        for node in out_degree_zeros:
            G.add_edge(node, "END", label=100)
        
        # Revert all edges
        G = G.reverse()
        
        # Delete redudant edges
        G_copy = copy.deepcopy(G)
        for current_node in G.nodes:
            for anchestor in G_copy.successors(current_node):
                G.remove_edge(current_node, anchestor)
                if not nx.has_path(G, current_node, anchestor):
                    G.add_edge(current_node, anchestor)

        return G, edge_labels