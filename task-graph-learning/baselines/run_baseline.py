# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import os
import click

try:
    from taskgraph.task_graph_learning import load_config_baseline
except:
    raise Exception("Taskgraph library not found. Please install it! Read the README.md file for more information.")

@click.command()
@click.option('--config', '-cfg', help='Path to the config file. You can find the config file in the config folder.', required=True)
def main(config:str):
    if not os.path.isfile(config):
        raise Exception(f"Config file '{config}' does not exist.")
    cfg = load_config_baseline(config)
    keysteps = cfg.KEYSTEPS_JSON_PATH 
    baseline_folder_output = cfg.OUTPUT_PATH 
    script = cfg.SCRIPT_PATH
    try:
        objective = cfg.OBJECTIVE
    except:
        objective = None
    try:
        max_length = cfg.MAX_LENGTH
    except:
        max_length = -1
    if not os.path.isfile(keysteps):
        raise Exception(f"Keysteps JSON file '{keysteps}' does not exist.")
    try:
        augmented = cfg.AUGMENTED
    except:
        augmented = False
    
    # Execute script
    if objective:
        if augmented:
            os.system(f"python {script} --keysteps {keysteps} --objective {objective} -bfo {baseline_folder_output} --augmentations --max_length {max_length}")
        else:
            os.system(f"python {script} --keysteps {keysteps} --objective {objective} -bfo {baseline_folder_output} --max_length {max_length}")
    else:
        os.system(f"python {script} --keysteps {keysteps} -bfo {baseline_folder_output} --max_length {max_length}")
    

if __name__ == '__main__':
    main()