import argparse
import pickle
import sys

from imblearn.over_sampling import RandomOverSampler

from sklearn.feature_selection import (
    SelectPercentile,
    mutual_info_classif,
)

import utils
import classifiers as clfs


def main(args):
    train_file = args.dataset_path + "_train.csv"
    dev_file = args.dataset_path + "_dev.csv"
    test_file = args.dataset_path + "_test.csv"

    print("Loading train/dev/test data-frames...")
    train_df = utils.load_dataset(train_file)
    dev_df = utils.load_dataset(dev_file)
    test_df = utils.load_dataset(test_file)

    train_X, train_y = utils.get_X_and_y(train_df)
    dev_X, dev_y = utils.get_X_and_y(dev_df)
    test_X, test_y = utils.get_X_and_y(test_df)

    # Perform feature selection based upon mutual information
    print("\nBeginning feature selection using mutual information criteria...")
    selector = SelectPercentile(mutual_info_classif, percentile=25)
    selector.fit_transform(train_X, train_y)
    good_features = selector.get_support(indices=True)

    # Select well performing features only for all datasets
    train_X = train_X.iloc[:, good_features]
    dev_X = dev_X.iloc[:, good_features]
    test_X = test_X.iloc[:, good_features]

    # Oversampling to adjust for the large class imbalance
    print("\nOversampling to correct for class imbalance...")
    ros = RandomOverSampler(random_state=17)
    balanced_trainX, balanced_train_Y = ros.fit_resample(train_X, train_y)

    print("\nFinished loading and prepping train/dev/test data...")

    train_and_save_classifier(
        clfs.LR_CLASSIFIER,
        clfs.LR_PARAMS,
        "logisitic_regression",
        balanced_trainX,
        balanced_train_Y,
        dev_X,
        dev_y,
    )

    train_and_save_classifier(
        clfs.RF_CLASSIFIER,
        clfs.RF_PARAMS,
        "random_forest",
        balanced_trainX,
        balanced_train_Y,
        dev_X,
        dev_y,
    )

    train_and_save_classifier(
        clfs.NB_CLASSIFIER,
        clfs.NB_PARAMS,
        "logisitic_regression",
        balanced_trainX,
        balanced_train_Y,
        dev_X,
        dev_y,
    )

    train_and_save_classifier(
        clfs.SVM_CLASSIFIER,
        clfs.SVM_PARAMS,
        "logisitic_regression",
        balanced_trainX,
        balanced_train_Y,
        dev_X,
        dev_y,
    )

    train_and_save_classifier(
        clfs.MLP_CLASSIFIER,
        clfs.MLP_PARAMS,
        "neural_net",
        balanced_trainX,
        balanced_train_Y,
        dev_X,
        dev_y,
    )


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
