import argparse

from sklearn.ensemble import RandomForestClassifier

import utils


def main(args):
    train_file = args.dataset_path + "_train.csv"
    dev_file = args.dataset_path + "_dev.csv"
    test_file = args.dataset_path + "_test.csv"

    train_df = utils.load_dataset(train_file)
    dev_df = utils.load_dataset(dev_file)
    test_df = utils.load_dataset(test_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the bank telemarketing classifier on a given dataset"
    )
    parser.add_argument(
        "-d",
        "--dataset_path",
        default="../data/bank-partial",
        help="base path to the dataset to use",
    )
    args = parser.parse_args()
    main(args)
