# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

# This script runs train_with_gt_text_unified.py for all yaml files in the CaptainCook4D folder.

import os
import threading
import click

@click.command()
def main():
    print("Running train_all_TGT-text.py")

    def run_training(yaml_file, device):
        print(f"Running training for {yaml_file}")
        command = f"python train_with_gt_text_unified.py --config ../../configs/CaptainCook4D-TGT-text/{yaml_file} --log --cuda {device} --exclude_current"
        os.system(command)
        print(f"Finished training for {yaml_file}")

    threads = []
    for yaml_file in os.listdir("../../configs/CaptainCook4D-TGT-text"):
        # Start a new thread for each yaml file
        # Max 2 threads at a time in device 0 and 1, change the device number if needed
        thread = threading.Thread(target=run_training, args=(yaml_file, len(threads)))
        threads.append(thread)
        thread.start()
        # Max 2 threads at a time in device 0 and 1
        if len(threads) >= 2:
            # Wait for the threads to complete
            for thread in threads:
                thread.join()
            threads = []

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()