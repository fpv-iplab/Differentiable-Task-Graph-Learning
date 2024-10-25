import json
import click
import pandas as pd
import os

def timecode_to_ms(timecode):
    # Split the timecode into its components
    hours, minutes, seconds, milliseconds = map(int, timecode.split(':'))
    
    # Calculate total milliseconds
    total_ms = ((hours * 3600) + (minutes * 60) + seconds) * 1000 + milliseconds
    return total_ms

def fix_annotations(root_file):
    with open(root_file, 'r') as f:
        contents = f.readlines()
    
    lines_to_modify = {
        "43":  0,
        "197": 5,
        "260": 4,
        "273": 4,
        "280": 4,
        "331": 4,
        "398": 4,
        "690": 4,
        "995": 4
    }
    lines_to_remove = [1,2,3,4,5,6,195]
    
    # Order the lines to remove
    lines_to_remove = sorted(lines_to_remove, reverse=True)
    
    # Modify the lines
    for line, value in lines_to_modify.items():
        true_line = contents[int(line)]
        true_line = true_line.split(",")
        new_line = ",".join(true_line[:-1])
        new_line = new_line + "," + str(value) + "\n"
        contents[int(line)] = new_line
        
    # Remove the lines
    for line in lines_to_remove:
        del contents[line]
    
    return contents

@click.command()
@click.option('--annotations', '-a', required=True, help='Path to the annotations file', default="./EPIC_Tent2019/Synchronised_action_label.txt")
@click.option('--split', '-s', required=True, help='Path to the split file')
@click.option('--dictionary', '-d', required=True, help='Path to the dictionary file', default="./EPIC-Tent-O_action_dict.json")
@click.option('--output', '-o', required=True, help='Path to the output file')
def main(annotations:str, split:str, dictionary:str, output:str):
    # Save in a new temporary file the fixed annotations
    with open(annotations.replace(".txt", "_fixed.txt"), 'w') as f:
        f.writelines(fix_annotations(annotations))
        
    annotations = annotations.replace(".txt", "_fixed.txt")
    
    # Load the split file
    split = json.load(open(split, 'r'))
    split_subject_id = [int(video.split("_")[-1]) for video in split]
    
    # Load the annotations
    annots = pd.read_csv(annotations, delimiter=",")
    
    # Delete the file
    os.remove(annotations)

    # Filter the annotations for the split_subject_id
    data = []
    for subject_id in split_subject_id:
        data.append(annots[annots["subject_id"] == subject_id])

    # Load the dictionary
    dictionary = json.load(open(dictionary, 'r'))

    annotations = {}

    for annot in data:
        video_name = "annotations_" + str(annot["subject_id"].values[0])
        segments = []
        count = 0
        old_idx = -1
        for _, row in annot.iterrows():
            start_time, end_time, idx = row["str_GoPro_ts"], row["end_GoPro_ts"], str(row["action_label"])
            if old_idx == int(idx):
                segments[-1]["end_time"] = timecode_to_ms(end_time)
                continue
            old_idx = int(idx)
            if count < split[video_name]:
                action = dictionary[str(int(idx)+1)]
                segments.append({
                    "step_id": int(idx),
                    "start_time": timecode_to_ms(start_time),
                    "end_time": timecode_to_ms(end_time),
                    "description": action,
                    "has_errors": False,
                    "step_name": action,
                })
            else:
                segments[-1]["has_errors"] = True
                break
            count += 1
        annotations[video_name] = {
            "scenario": "epic-tent",
            "recording_id": video_name,
            "segments": segments
        }

    taxonomy = {
        "epic-tent": {}
    }

    for k, v in dictionary.items():
        if int(k) == 0:
            continue
        taxonomy["epic-tent"][str(int(k)-1)] = {
            "name": v,
            "id": int(k)-1,
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