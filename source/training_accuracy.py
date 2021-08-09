import sys
import json

import matplotlib.pyplot as plt


def main():
    taining_data_path = sys.argv[1]
    training_fold_data = json.load(open(taining_data_path, "r"))

    plt.figure()
    plt.title("Training Accuracy on each cross validation fold")
    plt.xlabel("CV fold #")
    plt.ylabel("Accuracy")
    fold_indices = list(range(10))
    model_order = [
        "Logistic_Regression",
        "Random_Forest",
        "Naive_Bayes",
        "Linear_SVM",
        "Neural_Network",
    ]
    plt.xticks(fold_indices, [i + 1 for i in fold_indices])
    for model_name in model_order:
        acc_data = training_fold_data[model_name]
        plt.plot(
            fold_indices, acc_data, label=model_name.replace("_", " "), lw=2
        )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
