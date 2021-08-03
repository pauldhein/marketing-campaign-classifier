import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

import utils


def main(args):
    train_file = args.dataset_path + "_train.csv"
    dev_file = args.dataset_path + "_dev.csv"
    test_file = args.dataset_path + "_test.csv"

    print("Loading train/dev/test data-frames")
    train_df = utils.load_dataset(train_file)
    dev_df = utils.load_dataset(dev_file)
    test_df = utils.load_dataset(test_file)

    train_X, train_y = utils.get_X_and_y(train_df)
    dev_X, dev_y = utils.get_X_and_y(dev_df)
    test_X, test_y = utils.get_X_and_y(test_df)

    print("\nFinished loading and prepping train/dev/test data...")

    # NOTE: Using sensible defaults for now
    # NOTE: Using balanced class weighting to correct for the major class imbalance
    classifier = RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, class_weight="balanced"
    )

    print(f"\nBeginning classifier fit for:\n{str(classifier)}\n")
    classifier.fit(train_X, train_y)
    dev_preds = classifier.predict(dev_X)

    p = precision_score(dev_y, dev_preds)
    r = recall_score(dev_y, dev_preds)
    f = f1_score(dev_y, dev_preds)
    print(p, r, f)


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
