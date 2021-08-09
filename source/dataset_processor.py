import csv
import argparse

import pandas as pd
import numpy as np


def main(args):
    # Load the full dataset into a dataframe
    # NOTE: setting data types to ensure we get categorical types for each str value
    df = pd.read_csv(
        args.csv_data_path,
        sep=";",
        quotechar='"',
        dtype={
            "age": "int64",
            "job": "category",
            "marital": "category",
            "education": "category",
            "default": "category",
            "housing": "category",
            "loan": "category",
            "contact": "category",
            "month": "category",
            "day_of_week": "category",
            "duration": "int64",
            "campaign": "int64",
            "pdays": "int64",
            "previous": "int64",
            "poutcome": "category",
            "emp.var.rate": "float64",
            "cons.price.idx": "float64",
            "cons.conf.idx": "float64",
            "euribor3m": "float64",
            "nr.employed": "float64",
            "y": "category",
        },
    )
    print(df)

    # Transform label to integer encoding
    y_df = (df["y"]).to_frame().apply(lambda x: x.cat.codes)

    # Remove label from data and then transform categorical features to one-hot
    df.drop(columns="y", inplace=True)
    one_hot_df = pd.get_dummies(df)

    # Add labels back in to full dataset
    df = one_hot_df.join(y_df)
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
    save_dataset(train, args.csv_data_path, "_train.csv")
    save_dataset(dev, args.csv_data_path, "_dev.csv")
    save_dataset(test, args.csv_data_path, "_test.csv")


def save_dataset(dataset: pd.DataFrame, orig_path: str, file_ending: str):
    dataset_filepath = orig_path.replace(".csv", file_ending)
    dataset.to_csv(dataset_filepath, sep=";", quotechar='"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the UCI Bank Marketing Data Set for training/classification"
    )
    parser.add_argument(
        "-p",
        "--csv_data_path",
        default="../data/bank-full.csv",
        help="Base path to the dataset to use",
    )
    args = parser.parse_args()
    main(args)
