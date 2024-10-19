# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

# This scripts calculates the mean and confidence intervals of the results obtained from the experiments executed with different seeds.

import click
import json
import glob
import numpy as np
import random

def bootstrap(data, n_samples, n_iterations=1000):
    """Perform bootstrap resampling on the data and return confidence intervals."""
    means = []
    for _ in range(n_iterations):
        # Resample with replacement
        bootstrap_sample = [random.choice(data) for _ in range(n_samples)]
        # Compute the mean for the bootstrap sample
        means.append(np.mean(bootstrap_sample))
    
    # Get the 5th and 95th percentiles for the 90% confidence interval
    lower_bound = np.percentile(means, 5)
    upper_bound = np.percentile(means, 95)
    return lower_bound, upper_bound

@click.command()
@click.option('--results', '-r', help='Path to the results file')
def main(results: str):
    random.seed(42)
    folders = glob.glob(results + '/*')
    folder_confidence_intervals = {
        "Precision": [],
        "Recall": [],
        "F1": []
    }
    final_metrics = {
        "Precision": [],
        "Recall": [],
        "F1": []
    }

    for folder in folders:
        folder_metrics = {
            "Precision": [],
            "Recall": [],
            "F1": []
        }

        # Gather metrics for all JSON files in the folder
        json_files = glob.glob(folder + '/*.json')
        for file in json_files:
            with open(file) as f:
                data = json.load(f)
                folder_metrics["Precision"].append(data["Precision"])
                folder_metrics["Recall"].append(data["Recall"])
                folder_metrics["F1"].append(data["F1"])

        # Convert to numpy arrays for easier manipulation
        precision_array = np.array(folder_metrics["Precision"])
        recall_array = np.array(folder_metrics["Recall"])
        f1_array = np.array(folder_metrics["F1"])

        # Calculate mean for the folder and store in final metrics
        precision_mean = np.mean(precision_array)
        recall_mean = np.mean(recall_array)
        f1_mean = np.mean(f1_array)
        final_metrics["Precision"].append(precision_mean)
        final_metrics["Recall"].append(recall_mean)
        final_metrics["F1"].append(f1_mean)

        # Number of samples in this folder (n)
        n = len(json_files)
        
        precision_ci = bootstrap(folder_metrics["Precision"], n)
        precision_ci = precision_ci[1] - precision_ci[0]
        recall_ci = bootstrap(folder_metrics["Recall"], n)
        recall_ci = recall_ci[1] - recall_ci[0]
        f1_ci = bootstrap(folder_metrics["F1"], n)
        f1_ci = f1_ci[1] - f1_ci[0]

        # Store the confidence intervals for each folder
        folder_confidence_intervals["Precision"].append(precision_ci)
        folder_confidence_intervals["Recall"].append(recall_ci)
        folder_confidence_intervals["F1"].append(f1_ci)

    # Calculate the mean of the confidence intervals across all folders
    precision_ci_mean = np.mean(folder_confidence_intervals["Precision"])
    recall_ci_mean = np.mean(folder_confidence_intervals["Recall"])
    f1_ci_mean = np.mean(folder_confidence_intervals["F1"])

    # Calculate the final mean metrics across all folders
    final_precision_mean = np.mean(final_metrics["Precision"])
    final_recall_mean = np.mean(final_metrics["Recall"])
    final_f1_mean = np.mean(final_metrics["F1"])

    # Convert final metrics and confidence intervals to percentages
    final_precision_mean *= 100
    final_recall_mean *= 100
    final_f1_mean *= 100
    precision_ci_mean *= 100
    recall_ci_mean *= 100
    f1_ci_mean *= 100

    # Print the mean of the confidence intervals for each metric in percentages
    print(f"Mean 90% CI for Precision: {final_precision_mean:.2f}% ± {precision_ci_mean:.2f}%")
    print(f"Mean 90% CI for Recall: {final_recall_mean:.2f}% ± {recall_ci_mean:.2f}%")
    print(f"Mean 90% CI for F1: {final_f1_mean:.2f}% ± {f1_ci_mean:.2f}%")

if __name__ == '__main__':
    main()
