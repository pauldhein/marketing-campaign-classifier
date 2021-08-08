# Predicting Bank Term Deposits from Marketing Campaign Data

## Project Goal

Can we create a data processing pipeline and machine learning classifier that will predict whether a bank client will purchase a subscription to a bank term deposit (BTD)?

## Data source

This project makes use of the [UCI bank marketing data set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#) to train and evaluate the classifiers defined in this repository.

## Running the experiments

### Installation

1. Setup [Docker for desktop](https://www.docker.com/products/docker-desktop)
2. Pull the docker image for this repo: `docker pull pauldhein/telemarket-model:latest`

### Dataset preprocessing

This experiment ...

### Model training

This experiment ...

### Model evaluation

This experiment ...

### Observed metrics from the development set

| Model Name | Accuracy | Precision | Recall | F1 | AUC-ROC |
| ---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8593 | 0.4332 | 0.8644 | 0.5771 | 0.8616 |
| Random Forest       | **0.9034** | **0.5933** | 0.4125 | 0.4867 | 0.6886 |
| Naive Bayes         | 0.8334 | 0.3622 | 0.6574 | 0.4671 | 0.7564 |
| Linear SVM          | 0.2799 | 0.0498 | 0.3032 | 0.0855 | 0.2901 |
| Neural Network      | 0.8501 | 0.4212 | **0.9344** | **0.5806** | **0.8870** |

### Observed metrics from the test set

| Model Name | Accuracy | Precision | Recall | F1 | AUC-ROC |
| ---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8623 | 0.4324 | 0.8677 | 0.5772 | 0.8647 |
| Random Forest       | **0.9053** | **0.5870** | 0.4238 | 0.4922 | 0.6938 |
| Naive Bayes         | 0.8371 | 0.3616 | 0.6592 | 0.4670 | 0.7589 |
| Linear SVM          | 0.2765 | 0.0501 | 0.3161 | 0.0865 | 0.2939 |
| Neural Network      | 0.8555 | 0.4242 | **0.9350** | **0.5836** | **0.8904** |