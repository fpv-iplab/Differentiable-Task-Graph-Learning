# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import numpy as np
import torch
import click
import os

try:
    from taskgraph.task_graph_learning import TGT, load_config_task_graph_learning
except:
    raise Exception("You need to install the TGML library. Please read the README.md file.")

from sklearn.metrics import accuracy_score

# Set the environment variable for CUBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def rate(s,Z):
    mul = 1
    all = np.array(range(0,Z.shape[0]))
    for i in range(1,len(s)-1):
        J = s[:i] #past nodes
        notJ = np.array([x for x in all if x not in J]) #nodes
        current = s[i] #node
 
        num = (Z[current,:].sum() - Z[current,:][notJ].sum())
        
        den = 0
        for k in notJ:
            den = den + (Z[k,:].sum() - Z[k,:][notJ].sum())
        
        p = num/den if den != 0 else 0
        mul = mul * p
    return mul

@click.command()
@click.option("--config", "-cfg", type=str, required=True, help="Path to the config file. You can find the config file in the config folder.")
@click.option("--cuda", type=int, default=0, help="CUDA device to use.")
def main(config:str, cuda:int):
    
    # Load config
    cfg = load_config_task_graph_learning(config)

    # Activity name (e.g., "coffee")
    scenario = cfg.ACTIVITY_NAME
    
    # Path to the embeddings
    embeddings = cfg.EMBEDDINGS
        
    # Load annotations
    annotations = cfg.ANNOTATIONS

    # Select the device
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    gts = []
    preds = []

    # Create the model
    pre_trained = cfg.MODEL
    net = TGT(d_model=4096, device=device, dropout=0.25).to(device)
    net.load_state_dict(torch.load(pre_trained))
    print("Model loaded: ", pre_trained)

    gts = []
    preds = []

    for video in annotations["annotations"]:
        if annotations["annotations"][video]["scenario"] == scenario:
            scenario = annotations["annotations"][video]["scenario"]
            files = sorted(os.listdir(os.path.join(embeddings, scenario, video)), key=lambda x: int(x.split("_")[0]))
            dict_of_video = {}
            segments = annotations["annotations"][video]["segments"]
            for file in files:
                if file.endswith(".pt"):
                    key = int(file.split("_")[0])
                    if key not in dict_of_video:
                        dict_of_video[key] = {
                            "name": segments[key]["step_name"],
                            "video_embedding": torch.load(os.path.join(embeddings, scenario, video, file))["video"].view(-1),
                            "text_embedding": torch.load(os.path.join(embeddings, scenario, video, file))["text"].view(-1),
                        }
                
            i = 0
            correct_key_dict_of_video = {}
            for key in dict_of_video:
                correct_key_dict_of_video[i] = dict_of_video[key]
                i += 1
            
            for i in range(1, len(correct_key_dict_of_video) - 1):
                gts.append(1)
                video_embedding_1 = correct_key_dict_of_video[i-1]["video_embedding"]
                video_embedding_2 = correct_key_dict_of_video[i+1]["video_embedding"]

                video_embedding_analisys = torch.stack([video_embedding_1, video_embedding_2], dim=0)
                video_embedding_analisys = torch.nn.Embedding.from_pretrained(video_embedding_analisys, freeze=True)
                video_encoder_input = torch.nn.Sequential(video_embedding_analisys).to(device)   
                batch_size = torch.tensor(np.arange(2)).to(device)

                with torch.no_grad():
                    net.eval()
                    pred_adjacency_matrix, _ = net.get_adjacency_matrix(video_encoder_input(batch_size))
                    pred_adjacency_matrix = pred_adjacency_matrix.cpu().numpy()
                    
                # Take the matrix removing beta and gamma
                pred_adjacency_matrix = pred_adjacency_matrix[1:-1, 1:-1]
                
                sequence_true = [0, 1, 2, 3]
                sequence_false = [0, 2, 1, 3]
                if rate(sequence_true, pred_adjacency_matrix) > rate(sequence_false, pred_adjacency_matrix) or pred_adjacency_matrix[2, 1] > pred_adjacency_matrix[1, 2] or pred_adjacency_matrix[3, 2] > pred_adjacency_matrix[3, 1]:
                    preds.append(1)
                else:
                    preds.append(0)
        
    gts = np.array(gts)
    preds = np.array(preds)
    print("Scenario: ", scenario)
    print("GT:", gts)
    print("Preds:", preds)
    print("Accuracy: ", accuracy_score(gts, preds))
    
    os.makedirs("ordering", exist_ok=True)
    with open(f"ordering/{scenario}.txt", "w") as f:
        f.write(f"{accuracy_score(gts, preds)}")

if __name__ == "__main__":
    main()