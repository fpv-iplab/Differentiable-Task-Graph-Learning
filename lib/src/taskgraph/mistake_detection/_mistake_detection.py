import networkx as nx

def check_precondition(graph:nx.DiGraph, segments:list, video:str) -> tuple[dict, list]:
    """
    Description
    ------------
    Check the preconditions of the segments of the video.

    Parameters
    -----------
    - **graph (nx.DiGraph)**: Graph with the preconditions.
    - **segments (list)**: List of segments of the video.
    - **video (str)**: Video name.

    Returns
    --------
    - **tuple[dict, list]**: Results of the preconditions and predictions
    """
    results = {
        "video": video,
        "segments": []
    }
    predictions = []
    past = ["START"]
    for segment in segments:
        if str(segment["step_id"]) + "_" + segment["step_name"] not in graph.nodes or len(list(graph.predecessors(str(segment["step_id"]) + "_" + segment["step_name"]))) == 0:
            results["segments"].append({
                "step_id": segment["step_id"],
                "step_name": segment["step_name"],
                "preconditions": ["NOT DEFINED"],
                "mistake": []
            })
            predictions.append(1)
            continue
        else:
            preconditions = list(graph.predecessors(str(segment["step_id"]) + "_" + segment["step_name"]))
        if all([precondition in past for precondition in preconditions]):
            results["segments"].append({
                "step_id": segment["step_id"],
                "step_name": segment["step_name"],
                "preconditions": preconditions,
                "mistake": []
            })
            predictions.append(0)
        else:
            results["segments"].append({
                "step_id": segment["step_id"],
                "step_name": segment["step_name"],
                "preconditions": preconditions,
                "mistake": [precondition for precondition in preconditions if precondition not in past]
            })
            predictions.append(1)
        past.append(str(segment["step_id"]) + "_" + segment["step_name"])
    return results, predictions