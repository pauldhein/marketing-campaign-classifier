"""
This file holds a class hierarchy of classifier wrapper classes around common Scikit-learn classifiers. The purpose of these wrappers is to ensure that all classifiers have:
    (1) a well defined natural language name for printing
    (2) methods to allow the saved configuration of a classifier to be saved/loaded to/from a Python pickle file
    (3) specified parameter grids that will be used for hyperparameter tuning with the Scikit-learn GridSearchCV method
"""

from abc import ABC, abstractclassmethod, abstractstaticmethod
from dataclasses import dataclass
from typing import Callable, Dict, List
import pickle
import os

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier


@dataclass
class AbstractClassifier(ABC):
    name: str
    model: ClassifierMixin

    @abstractstaticmethod
    def classifier_name():
        return NotImplemented

    @abstractstaticmethod
    def get_param_grid():
        return NotImplemented

    @abstractclassmethod
    def from_scratch(cls):
        return NotImplemented

    @abstractclassmethod
    def from_file(cls, filepath: str):
        return NotImplemented

    @staticmethod
    def full_classifier_path(path: str, clf_name: str) -> str:
        filename = f"trained_{clf_name}_classifier.pkl"
        return os.path.normpath(os.path.join(path, filename))

    @classmethod
    def load_saved_classifier(
        cls, filepath: str, clf_name: str
    ) -> ClassifierMixin:
        full_path = cls.full_classifier_path(filepath, clf_name)
        return pickle.load(open(full_path, "rb"))

    def to_file(self, model: ClassifierMixin, filepath: str):
        full_path = self.full_classifier_path(filepath, self.name)
        pickle.dump(model, open(full_path, "wb"))


@dataclass
class LogRegClassifier(AbstractClassifier):
    @staticmethod
    def get_param_grid():
        return {
            "classifier__penalty": ["l1", "l2"],
            "classifier__C": [0.1, 0.5, 1.0, 5.0, 10.0],
        }

    @staticmethod
    def classifier_name():
        return "Logistic_Regression"

    @classmethod
    def from_file(cls, filepath: str):
        name = cls.classifier_name()
        return cls(name, cls.load_saved_classifier(filepath, name))

    @classmethod
    def from_scratch(cls):
        return cls(
            cls.classifier_name(),
            LogisticRegression(
                solver="liblinear", max_iter=200, random_state=17
            ),
        )


@dataclass
class RandForestClassifier(AbstractClassifier):
    @staticmethod
    def get_param_grid():
        return {
            "classifier__max_features": [2, 5, 10, 15, 20],
            "classifier__n_estimators": [2, 5, 10, 20, 50, 100],
        }

    @staticmethod
    def classifier_name():
        return "Random_Forest"

    @classmethod
    def from_file(cls, filepath: str):
        name = cls.classifier_name()
        return cls(name, cls.load_saved_classifier(filepath, name))

    @classmethod
    def from_scratch(cls):
        return cls(
            cls.classifier_name(),
            RandomForestClassifier(
                max_depth=None,
                min_samples_split=2,
            ),
        )


@dataclass
class NaiveBayesClassifier(AbstractClassifier):
    @staticmethod
    def get_param_grid():
        return {}

    @staticmethod
    def classifier_name():
        return "Naive_Bayes"

    @classmethod
    def from_file(cls, filepath: str):
        name = cls.classifier_name()
        return cls(name, cls.load_saved_classifier(filepath, name))

    @classmethod
    def from_scratch(cls):
        return cls(cls.classifier_name(), GaussianNB())


@dataclass
class SVMClassifier(AbstractClassifier):
    @staticmethod
    def get_param_grid():
        return {
            "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0],
        }

    @staticmethod
    def classifier_name():
        return "Linear_SVM"

    @classmethod
    def from_file(cls, filepath: str):
        name = cls.classifier_name()
        return cls(name, cls.load_saved_classifier(filepath, name))

    @classmethod
    def from_scratch(cls):
        return cls(
            cls.classifier_name(),
            SVC(
                kernel="linear",
                probability=True,
                max_iter=1000,
                random_state=17,
            ),
        )


@dataclass
class NeuralNetClassifier(AbstractClassifier):
    @staticmethod
    def get_param_grid():
        return {
            "classifier__hidden_layer_sizes": [
                (10,),
                (20,),
                (10, 10),
                (10, 5),
                (20, 10),
            ],
            "classifier__alpha": [0.01, 0.1, 1.0],
        }

    @staticmethod
    def classifier_name():
        return "Neural_Network"

    @classmethod
    def from_file(cls, filepath: str):
        name = cls.classifier_name()
        return cls(name, cls.load_saved_classifier(filepath, name))

    @classmethod
    def from_scratch(cls):
        return cls(
            cls.classifier_name(),
            MLPClassifier(
                max_iter=500,
                random_state=17,
            ),
        )
