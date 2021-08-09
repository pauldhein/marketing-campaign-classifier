import os
import json
import argparse
from typing import List, Tuple

from pandas import DataFrame
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
)

import utils
import classifiers as clfs


DATA_PATH = "../data/"


def main(args):
    train_file = args.dataset_path + "_train.csv"
    dev_file = args.dataset_path + "_dev.csv"
    test_file = args.dataset_path + "_test.csv"

    print("Loading train/dev/test data-frames...")
    train_X, train_y = utils.load_Xy_dataset(train_file)
    dev_X, dev_y = utils.load_Xy_dataset(dev_file)
    test_X, test_y = utils.load_Xy_dataset(test_file)

    # Perform feature selection based upon mutual information
    print("\nBeginning feature selection using mutual information criteria...")
    selector = SelectKBest(mutual_info_classif, k=20)
    selector.fit_transform(train_X, train_y)
    good_features = selector.get_support(indices=True)
    print(f"Number of features to use: {len(good_features)}")
    print(f"Feature column names:\n{train_X.iloc[:, good_features].columns}")

    # Select well performing features only for all datasets
    train_X = train_X.iloc[:, good_features]
    dev_X = dev_X.iloc[:, good_features]
    test_X = test_X.iloc[:, good_features]

    # Oversampling to adjust for the large class imbalance
    print("\nOversampling to correct for class imbalance...")
    ros = RandomOverSampler(random_state=17)
    train_X, train_Y = ros.fit_resample(train_X, train_y)

    print("\nFinished loading and prepping train/dev/test data...")
    lr_clf = clfs.LogRegClassifier.from_scratch()
    (fitted_lr_clf, lr_training_scores) = train_and_score_classifier(
        lr_clf, train_X, train_Y, dev_X, dev_y
    )
    lr_clf.to_file(fitted_lr_clf, DATA_PATH)

    rf_clf = clfs.RandForestClassifier.from_scratch()
    (fitted_rf_clf, rf_training_scores) = train_and_score_classifier(
        rf_clf, train_X, train_Y, dev_X, dev_y
    )
    rf_clf.to_file(fitted_rf_clf, DATA_PATH)

    nb_clf = clfs.NaiveBayesClassifier.from_scratch()
    (fitted_nb_clf, nb_training_scores) = train_and_score_classifier(
        nb_clf, train_X, train_Y, dev_X, dev_y
    )
    nb_clf.to_file(fitted_nb_clf, DATA_PATH)

    svm_clf = clfs.SVMClassifier.from_scratch()
    (fitted_svm_clf, svm_training_scores) = train_and_score_classifier(
        svm_clf, train_X, train_Y, dev_X, dev_y
    )
    svm_clf.to_file(fitted_svm_clf, DATA_PATH)

    nn_clf = clfs.NeuralNetClassifier.from_scratch()
    (fitted_nn_clf, nn_training_scores) = train_and_score_classifier(
        nn_clf, train_X, train_Y, dev_X, dev_y
    )
    nn_clf.to_file(fitted_nn_clf, DATA_PATH)

    # Save CV fold data from model training
    training_cv_data = {
        "Logistic_Regression": lr_training_scores,
        "Random_Forest": rf_training_scores,
        "Neural_Network": nn_training_scores,
        "Naive_Bayes": nb_training_scores,
        "Linear_SVM": svm_training_scores,
    }
    training_data_path = os.path.normpath(
        os.path.join(DATA_PATH, "training_fold_data.json")
    )
    json.dump(training_cv_data, open(training_data_path, "w"))


def train_and_score_classifier(
    classifier: clfs.AbstractClassifier,
    train_X: DataFrame,
    train_y: DataFrame,
    dev_X: DataFrame,
    dev_y: DataFrame,
) -> Tuple[ClassifierMixin, List[float]]:
    """Runs grid search via 10-fold cross validation for the parameter grids specified in the `param_grid` method of the `classifier` object. The best choice fitted model is then evaluated with a set of standard metrics and returned along with the training accuracies for each fold of cross validation.

    Args:
        classifier (clfs.AbstractClassifier): a classifier to be fitted after GridSearch
        train_X (DataFrame): the training feature data
        train_y (DataFrame): the training label data
        dev_X (DataFrame): the developement feature data
        dev_y (DataFrame): the development label data

    Returns:
        Tuple[ClassifierMixin, List[float]]: the best fit classifier and its associated training accuracy per fold of the 10 CV folds
    """
    # Adding feature scaling into the classifier pipeline
    clf_pipeline = GridSearchCV(
        Pipeline(
            steps=[
                ("scaling", MinMaxScaler()),
                ("classifier", classifier.model),
            ]
        ),
        classifier.get_param_grid(),
        cv=10,
        verbose=1,
        n_jobs=-1,
        return_train_score=True,
    )

    print(f"\nBeginning classifier fit for:\n{str(clf_pipeline)}\n")
    clf_pipeline.fit(train_X, train_y)

    print(f"Scores for {classifier.name} classifier")
    best_clf = clf_pipeline.best_estimator_
    dev_preds = best_clf.predict(dev_X)
    utils.compute_metrics(dev_y, dev_preds)

    best_idx = clf_pipeline.best_index_
    training_fold_accs = [
        clf_pipeline.cv_results_[f"split{i}_train_score"][best_idx]
        for i in range(10)
    ]
    return (best_clf, training_fold_accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the bank telemarketing classifiers on a given dataset"
    )
    parser.add_argument(
        "-p",
        "--dataset_path",
        default="../data/bank-partial",
        help="Base path to the dataset to use",
    )
    args = parser.parse_args()
    main(args)
