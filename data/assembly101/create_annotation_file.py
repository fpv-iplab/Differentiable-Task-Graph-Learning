import json
import click
import pandas as pd
from glob import glob

@click.command()
@click.option('--annotations', '-a', required=True, help='Path to the annotations folder', default="./assembly101-mistake-detection/annots")
@click.option('--split', '-s', required=True, help='Path to the split file')
@click.option('--dictionary', '-d', required=True, help='Path to the dictionary file', default="./Assembly101-O_action_dict.json")
@click.option('--output', '-o', required=True, help='Path to the output file')
def main(annotations:str, split:str, dictionary:str, output:str):
    # Load the split file
    with open(split, 'r') as f:
        split = f.readlines()
    split = [s.strip() for s in split]

    # Load the annotations
    annots = glob(annotations + "/*.csv")

    # Filter the annotations for the split
    data = [annot for annot in annots if annot.split("/")[-1].split(".")[0] in split]

    # Load the dictionary
    dictionary = json.load(open(dictionary, 'r'))

    # Reverse the dictionary
    reverse_dict = {v: k for k, v in dictionary.items()}

    annotations = {}

    for annot in data:
        df = pd.read_csv(annot, header=None)
        video_name = annot.split("/")[-1].split(".")[0]
        segments = []
        for _, row in df.iterrows():
            start_time, end_time, verb, this, correction = row[0], row[1], row[2], row[3], row[5]
            action = verb + "-" + this.replace(" ", "_")
            if action == "position-figurine":
                action = "attach-figurine"
            idx = reverse_dict[action]
            if correction != "mistake":
                idx = reverse_dict[action]
                segments.append({
                    "step_id": int(idx),
                    "start_time": start_time,
                    "end_time": end_time,
                    "description": action,
                    "has_errors": False,
                    "step_name": action,
                })
            else:
                idx = reverse_dict[action]
                segments.append({
                    "step_id": int(idx),
                    "start_time": start_time,
                    "end_time": end_time,
                    "description": action,
                    "has_errors": True,
                    "step_name": action,
                })
                break
        annotations[video_name] = {
            "scenario": "assembly101",
            "recording_id": video_name,
            "segments": segments
        }

    taxonomy = {
        "assembly101": {}
    }

    for k, v in dictionary.items():
        taxonomy["assembly101"][k] = {
            "name": v,
            "id": int(k),
            "is_leafnode": True,
            "parent_id": None,
            "parent_name": None
        }

    annotations_taxonomy = {
        "annotations": annotations,
        "taxonomy": taxonomy
    }

    json.dump(annotations_taxonomy, open(output, 'w'), indent=4)

if __name__ == "__main__":
    main()