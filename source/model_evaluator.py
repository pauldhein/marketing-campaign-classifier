import os
import argparse

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
)
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

import classifiers as clfs
import utils


def main(args):
    print("Loading and preparing datasets...")
    # Get the selected features from our training data
    train_X, train_y, good_features = recover_good_features(args.data_path)

    # Prepare all of the classifiers
    lr = clfs.LogRegClassifier.from_file(args.data_path)
    rf = clfs.RandForestClassifier.from_file(args.data_path)
    nb = clfs.NaiveBayesClassifier.from_file(args.data_path)
    svm = clfs.SVMClassifier.from_file(args.data_path)
    nn = clfs.NeuralNetClassifier.from_file(args.data_path)

    print("Fitting saved model configs...")
    # Re-fitting the saved model configurations
    lr.model.fit(train_X, train_y)
    rf.model.fit(train_X, train_y)
    nb.model.fit(train_X, train_y)
    svm.model.fit(train_X, train_y)
    nn.model.fit(train_X, train_y)

    if args.dev:
        print("Evaluating on the DEV set")
        dev_file = args.data_path + "bank-full_dev.csv"
        dev_X, dev_y = load_eval_dataset(dev_file, good_features)

        # Perform metric evaluation and retrieve ROC curve data
        lr_fpr, lr_tpr = evaluate_classifier(lr, dev_X, dev_y)
        rf_fpr, rf_tpr = evaluate_classifier(rf, dev_X, dev_y)
        nb_fpr, nb_tpr = evaluate_classifier(nb, dev_X, dev_y)
        svm_fpr, svm_tpr = evaluate_classifier(svm, dev_X, dev_y)
        nn_fpr, nn_tpr = evaluate_classifier(nn, dev_X, dev_y)

        # Plot the ROC curves for all models
        roc_plot(
            "development",
            [
                (lr, lr_fpr, lr_tpr),
                (rf, rf_fpr, rf_tpr),
                (nb, nb_fpr, nb_tpr),
                (svm, svm_fpr, svm_tpr),
                (nn, nn_fpr, nn_tpr),
            ],
        )

    if args.test:
        print("Evaluating on the TEST set")
        test_file = args.data_path + "bank-full_test.csv"
        test_X, test_y = load_eval_dataset(test_file, good_features)

        # Perform metric evaluation and retrieve ROC curve data
        lr_fpr, lr_tpr = evaluate_classifier(lr, test_X, test_y)
        rf_fpr, rf_tpr = evaluate_classifier(rf, test_X, test_y)
        nb_fpr, nb_tpr = evaluate_classifier(nb, test_X, test_y)
        svm_fpr, svm_tpr = evaluate_classifier(svm, test_X, test_y)
        nn_fpr, nn_tpr = evaluate_classifier(nn, test_X, test_y)

        # Plot the ROC curves for all models
        roc_plot(
            "test",
            [
                (lr, lr_fpr, lr_tpr),
                (rf, rf_fpr, rf_tpr),
                (nb, nb_fpr, nb_tpr),
                (svm, svm_fpr, svm_tpr),
                (nn, nn_fpr, nn_tpr),
            ],
        )

    plt.show()


def evaluate_classifier(classifier, X, y):
    """Compute predictions for this classifier and then use those predictions (and prediction probabilities) to compute all key metrics and the ROC curve.

    Args:
        classifier (AbstractClassifier): the classifier to be evaluated
        X (DataFrame): the feature dat to be used for evaluation
        y (DataFrame): the label data to be used for evaluating predictions

    Returns:
        Tuple[List[float], List[float]]: The FPR and TPR values of the ROC curve
    """
    print(f"Evaluating {classifier.name.replace('_', ' ')}")
    preds = classifier.model.predict(X)
    utils.compute_metrics(y, preds)
    probs = classifier.model.predict_proba(X)
    fpr, tpr, _ = roc_curve(y, probs[:, 1], drop_intermediate=False)
    return fpr, tpr


def load_eval_dataset(data_filepath, selected_features):
    """Loads a dataset to be used for evaluation. The dataset is reformed to only use the selected features from KBest selection.

    Args:
        data_filepath (str): path to the dataset to be loaded
        selected_features (ndarray): the indices of features to be selected from the dataset DataFrame

    Returns:
        Tuple[DataFrame, DataFrame]: The features and label DataFrames for this evaluation dataset
    """
    X, y = utils.load_Xy_dataset(data_filepath)
    X = X.iloc[:, selected_features]
    return X, y


def recover_good_features(data_path):
    """Returns the selected features from KBest selection and train datasets to refit the saved model configs.

    Args:
        data_path (str): The base path to the training dataset

    Returns:
        Tuple[DataFrame, DataFrame, ndarray]: All data needed to refit and evaluate with the dev or test datasets
    """
    train_file = data_path + "bank-full_train.csv"
    train_X, train_y = utils.load_Xy_dataset(train_file)

    selector = SelectKBest(mutual_info_classif, k=20)
    selector.fit_transform(train_X, train_y)
    good_features = selector.get_support(indices=True)

    train_X = train_X.iloc[:, good_features]

    ros = RandomOverSampler(random_state=17)
    train_X, train_y = ros.fit_resample(train_X, train_y)
    return train_X, train_y, good_features


def roc_plot(dataset_id, classifier_data):
    """Generates an ROC curve plot for classifier evaluation data on some dataset.

    NOTE: this function also plots the line y=x for easy interpretation of the ROC curve results.

    Args:
        dataset_id (str): a descriptive name of the dataset used to compute the current ROC curve data
        classifier_data (Tuple[AbstractClassifier, ndarray, ndarray]): the classifier that has been evaluated along with the TPR/FPR values of the ROC curve to be plotted
    """
    plt.figure()
    plt.title(f"ROC Curves for the {dataset_id} dataset")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    lw = 2
    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    for (clf, fpr, tpr) in classifier_data:
        plt.plot(fpr, tpr, lw=lw, label=clf.name.replace("_", " "))

    plt.legend(loc="lower right")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the bank telemarketing classifiers on a given dataset"
    )
    parser.add_argument(
        "-d",
        "--dev",
        action="store_true",
        help="Perform evaluation on the development dataset",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Perform evaluation on the test dataset",
    )
    parser.add_argument(
        "-p",
        "--data_path",
        default="../data/",
        help="Base path to the dataset to use",
    )
    args = parser.parse_args()
    main(args)
