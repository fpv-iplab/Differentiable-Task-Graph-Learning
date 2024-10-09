# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

# This script runs train_with_gt.py for all yaml files in the CaptainCook4D folder.

import os
import threading
import click

@click.command()
@click.option("--more_seeds", is_flag=True, help="Use multiple seeds for error bars.")
@click.option("--device", "-d", type=str, default="cuda:0", help="Device to use for training.")
def main(more_seeds:bool, device:str):
    print("more_seeds:", more_seeds)
    print("Running train_all_with_gt.py")

    def run_training(yaml_file):
        print(f"Running training for {yaml_file}")
        if more_seeds:
            # Use three different seeds for error bars
            for seed in [42, 1337, 2024, 2025, 2026]:
                command = f"python train_with_gt.py --config ../../configs/CaptainCook4D-DO/{yaml_file} --seed {seed} --log --device {device}"
                os.system(command)
        else:
            command = f"python train_with_gt.py --config ../../configs/CaptainCook4D-DO/{yaml_file} --log --device {device}"
            os.system(command)
        print(f"Finished training for {yaml_file}")

    threads = []
    for yaml_file in os.listdir("../../configs/CaptainCook4D-DO"):
        # Start a new thread for each yaml file
        thread = threading.Thread(target=run_training, args=(yaml_file,))
        threads.append(thread)
        thread.start()
        # Max 8 threads at a time
        if len(threads) >= 8:
            # Wait for the threads to complete
            for thread in threads:
                thread.join()
            threads = []

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
