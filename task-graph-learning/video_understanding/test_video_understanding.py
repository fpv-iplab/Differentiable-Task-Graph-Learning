# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

import os
import threading
import click

@click.command()
@click.option("--forecasting", type=bool, is_flag=True, default=False, help="False for ordering, True for forecasting.")
def main(forecasting:bool):

    def run_training(yaml_file, device, forecasting):
        if not forecasting:
            print(f"Running ordering for {yaml_file}")
            command = f"python ordering.py -cfg ../../configs/CaptainCook4D-Video-Understanding/{yaml_file} --cuda {device}"
        else:
            print(f"Running forecasting for {yaml_file}")
            command = f"python forecasting.py -cfg ../../configs/CaptainCook4D-Video-Understanding/{yaml_file} --cuda {device}"
        os.system(command)

    threads = []
    for yaml_file in os.listdir("../../configs/CaptainCook4D-Video-Understanding"):
        # Start a new thread for each yaml file
        # Max 2 threads at a time in device 0 and 1, change the device number if needed
        thread = threading.Thread(target=run_training, args=(yaml_file, len(threads), forecasting))
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
