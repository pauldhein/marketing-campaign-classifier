from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

LR_CLASSIFIER = LogisticRegression(
    solver="liblinear", max_iter=200, random_state=17
)
LR_PARAMS = {
    "classifier__penalty": ["l1", "l2"],
    "classifier__C": [0.1, 0.5, 1.0, 5.0, 10.0],
}

RF_CLASSIFIER = RandomForestClassifier(
    max_depth=None,
    min_samples_split=2,
)
RF_PARAMS = {
    "classifier__max_features": [1, 2, 5, 10, 15],
    "classifier__n_estimators": [2, 5, 10, 20, 50, 100],
}

MLP_CLASSIFIER = MLPClassifier(
    max_iter=500,
    random_state=17,
)
MLP_PARAMS = {
    "classifier__hidden_layer_sizes": [
        (10,),
        (20,),
        (10, 10),
        (10, 5),
        (20, 10),
    ],
    "classifier__alpha": [0.01, 0.1, 1.0],
}

BEST_MLP_CLASSIFIER = MLPClassifier(
    alpha=0.01,
    hidden_layer_sizes=(10, 10),
    random_state=17,
    max_iter=200,
)

NB_CLASSIFIER = GaussianNB()
NB_PARAMS = {}

SVM_CLASSIFIER = SVC(kernel="rbf", max_iter=500, random_state=17)
SVM_PARAMS = {
    "classifier__C": [0.1, 0.5, 1.0, 2.0, 5.0],
    "classifier__gamma": [0.1, 0.5, 1.0, 5.0, 10.0],
}
