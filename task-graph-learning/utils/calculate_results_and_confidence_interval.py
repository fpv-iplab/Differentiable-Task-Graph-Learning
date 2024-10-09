# Copyright (c) FPV@IPLab, and its affiliates. All Rights Reserved.

# This scripts calculates the mean and confidence intervals of the results obtained from the experiments executed with different seeds.

import click
import json
import glob
import numpy as np

@click.command()
@click.option('--results', '-r', help='Path to the results file')
def main(results: str):
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

    # Confidence value for 95% CI with t-distribution
    confidence_value = 2.776

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

        # Calculate standard deviation for the folder
        precision_std = np.std(precision_array, ddof=1)
        recall_std = np.std(recall_array, ddof=1)
        f1_std = np.std(f1_array, ddof=1)

        # Number of samples in this folder (n)
        n = len(json_files)

        # Calculate confidence intervals for this folder
        precision_ci = confidence_value * (precision_std / np.sqrt(n))
        recall_ci = confidence_value * (recall_std / np.sqrt(n))
        f1_ci = confidence_value * (f1_std / np.sqrt(n))

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
    print(f"Mean 95% CI for Precision: {final_precision_mean:.2f}% ± {precision_ci_mean:.2f}%")
    print(f"Mean 95% CI for Recall: {final_recall_mean:.2f}% ± {recall_ci_mean:.2f}%")
    print(f"Mean 95% CI for F1: {final_f1_mean:.2f}% ± {f1_ci_mean:.2f}%")

if __name__ == '__main__':
    main()
