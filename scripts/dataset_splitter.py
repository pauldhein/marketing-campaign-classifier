import csv
import sys

import pandas as pd
import numpy as np


def main():
    if len(sys.argv) != 2:
        raise RuntimeError(
            f"Program requires exaclty one input: path to a CSV dataset"
        )
    # Load the full dataset into a dataframe
    csv_data_path = sys.argv[1]
    df = pd.read_csv(csv_data_path, sep=";", quotechar='"')
    print(df)

    # Shuffle and split the dataset into train, dev, and test folds
    # following a (75%, 15%, 10%) pattern
    num_examples = len(df)
    amt_train = 0.75
    amt_train_and_Dev = 0.9
    train, dev, test = np.split(
        df.sample(frac=1, random_state=17),
        [int(amt_train * num_examples), int(amt_train_and_Dev * num_examples)],
    )
    print(train)
    print(len(train), len(dev), len(test))

    # Output the split datasets to train, dev, test CSV files
    csv_train_data_path = csv_data_path.replace(".csv", "_train.csv")
    csv_dev_data_path = csv_data_path.replace(".csv", "_dev.csv")
    csv_test_data_path = csv_data_path.replace(".csv", "_test.csv")
    train.to_csv(csv_train_data_path, sep=";", quotechar='"')
    dev.to_csv(csv_dev_data_path, sep=";", quotechar='"')
    test.to_csv(csv_test_data_path, sep=";", quotechar='"')


if __name__ == "__main__":
    main()
