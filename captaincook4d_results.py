# This script is used to calculate the mean and standard deviation of the results obtained from the experiments
# It also generate some json files with the results obtained (you can compare them with Table 1 of the paper).

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    tgl = "./Experiments"

    tgl_list = os.listdir(tgl)
    tgl_list = sorted(tgl_list)

    if len(tgl_list) < 24:
        print("There are missing experiments")
        print("Plese run: python train_all.py --more_seeds")
        return

    output_results = {}

    for folder in tgl_list:
        scenario = folder
        output_results[scenario] = {}
        file = os.path.join(tgl, folder)
        for file in os.listdir(file):
            if file.endswith(".json"):
                output_results[scenario][file.split(".")[0] + "_tgl"] = json.load(open(os.path.join(tgl, folder, file)))
        
    keys = list(output_results["blenderbananapancakes"].keys())

    output_results_mean = {}
    for key in keys:
        output_results_mean[key] = {
            "Precision": 0,
            "Recall": 0,
            "F1": 0
        }
        for scenario in output_results.keys():
            output_results_mean[key]["Precision"] += output_results[scenario][key]["Precision"]
            output_results_mean[key]["Recall"] += output_results[scenario][key]["Recall"]
            output_results_mean[key]["F1"] += output_results[scenario][key]["F1"]

        output_results_mean[key]["Precision"] /= len(output_results.keys())
        output_results_mean[key]["Recall"] /= len(output_results.keys())
        output_results_mean[key]["F1"] /= len(output_results.keys())
    
    # Take all the metrics for tgl
    tgl_metrics = []
    for key in output_results_mean.keys():
        tgl_metrics.append(
            [
                output_results_mean[key]["Precision"],
                output_results_mean[key]["Recall"],
                output_results_mean[key]["F1"]
            ]
        )

    tgl_metrics = np.array(tgl_metrics)

    mean_errors = np.mean(tgl_metrics, axis=0)

    # 2.776 -> 95% confidence interval
    std_errors = ((np.std(tgl_metrics, ddof=1, axis=0) / np.sqrt(len(tgl_metrics))) * 2.776)

    x = np.arange(len(mean_errors))

    os.makedirs("./Table_1", exist_ok=True)

    plt.bar(x, mean_errors, yerr=std_errors, capsize=5)
    plt.xlabel('Error Metric Index')
    plt.ylabel('Error Value')
    plt.title('Mean Errors with Standard Deviation')
    plt.xticks(x, ['Precision', 'Recall', 'F1'])
    plt.savefig("./Table_1/captaincook4d_errors_tgl.png")
    plt.show()
    
    std_errors = std_errors * 100
    # Save the errors in a JSON file
    errors = {
        "Precision": std_errors[0],
        "Recall": std_errors[1],
        "F1": std_errors[2]
    }

    means = {
        "Precision": mean_errors[0],
        "Recall": mean_errors[1],
        "F1": mean_errors[2]
    }

    # Create JSON file with the mean results using DO
    json.dump(means, open("./Table_1/captaincook4d_means_do.json", "w"), indent=4)

    # Create JSON file with the errors using DO
    json.dump(errors, open("./Table_1/captaincook4d_errors_do.json", "w"), indent=4)

    # Create JSON file with the mean results
    json.dump(output_results_mean, open("./Table_1/captaincook4d_results_mean.json", "w"), indent=4)
    
    # Create a JSON file with the results
    json.dump(output_results, open("./Table_1/captaincook4d_results.json", "w"), indent=4)

if __name__ == "__main__":
    main()