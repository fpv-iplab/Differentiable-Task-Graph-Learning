import os
import threading
import click

@click.command()
@click.option("--pre_trained", type=str, required=True, help="Path to the pre-trained model.")
def main(pre_trained:str):
    print("Running test_all_generation.py")

    def run_testing(yaml_file):
        print(f"Running test for {yaml_file}")
        command = f"python test_generation.py --config ../../configs/CaptainCook4D-TGT-text/{yaml_file} --pre_trained {pre_trained}"
        os.system(command)
        print(f"Finished test for {yaml_file}")

    threads = []
    for yaml_file in os.listdir("../../configs/CaptainCook4D-TGT-text"):
        # Start a new thread for each yaml file
        thread = threading.Thread(target=run_testing, args=(yaml_file,))
        threads.append(thread)
        thread.start()
        if len(threads) >= 1:
            # Wait for the threads to complete
            for thread in threads:
                thread.join()
            threads = []

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
