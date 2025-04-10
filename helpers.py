import os
import time
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler, \
    FunctionTransformer  # , OrdinalEncoder
# from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix,
    average_precision_score
)

from csv_parser import create_patient_summaries
from text_emb_aggregator import EmbeddingAggregator


# Load features
def load_features(file_path, delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter)
    print(f"features: {data}")
    return data


# Load labels
def load_labels(file_path="y.csv", delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter)
    return np.array(data.values.ravel())


# Load features as text summaries (create if doesn't exist)
def load_summaries(file_name):
    if not os.path.exists(file_name):
        print("File not found")
    with open(file_name, "r") as file:
        summaries_list = [line.strip() for line in file.readlines()]
    return summaries_list


def logistic_regression(dataset_name, X, y, nominal_features, n_repeats=10, n_splits=3, n_components=None):
    # for csv format
    dataset = dataset_name
    ml_method = "logistic regression"
    emb_method = "none"
    pca_components = f"PCA ({n_components} components)" if n_components else "none"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)

    pca_step = ("pca", PCA(n_components=n_components)) if n_components else None
    numerical_features = list(set(X.columns.values) - set(nominal_features))

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), nominal_features),
                ("numerical", Pipeline([
                                           ("numerical_imputer", IterativeImputer(max_iter=50)),
                                           ("numerical_scaler", MinMaxScaler())
                                       ] + ([pca_step] if pca_step else [])), numerical_features),
            ])),
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
        ]),
        param_grid={"classifier__C": [2, 10]},
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=n_repeats)
    )

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f"Log reg test fitting... ")
        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_per_fold.append({
            "Fold": len(metrics_per_fold),
            "AUC": roc_auc_score(y_test, y_test_pred_proba),
            "AP": average_precision_score(y_test, y_test_pred_proba),
            "Sensitivity": recall_score(y_test, y_test_pred, pos_label=1),
            "Specificity": specificity,
            "Precision": precision_score(y_test, y_test_pred, zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average='macro'),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred)
        })

    print(f"Log reg train fitting... ")
    search.fit(X, y)
    y_train_pred = search.predict(X)
    y_train_pred_proba = search.predict_proba(X)[:, 1]

    # Training metrics
    train_metrics = {
        "AUC": roc_auc_score(y, y_train_pred_proba),
        "AP": average_precision_score(y, y_train_pred_proba),
        "Sensitivity": recall_score(y, y_train_pred, pos_label=1),
        "Specificity": confusion_matrix(y, y_train_pred).ravel()[0] / (
                confusion_matrix(y, y_train_pred).ravel()[0] + confusion_matrix(y, y_train_pred).ravel()[1]),
        "Precision": precision_score(y, y_train_pred, zero_division=0),
        "F1": f1_score(y, y_train_pred, average='macro'),
        "Balanced Accuracy": balanced_accuracy_score(y, y_train_pred)
    }

    best_params = f"{search.best_params_}"

    print(f"Feature set size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, best_params, pca_components, train_metrics, metrics_per_fold


def lr_rte(dataset_name, X, y, nominal_features, n_splits=3, n_components=None):
    dataset = dataset_name
    ml_method = "logistic regression"
    emb_method = "RTE"
    concatenation = "no"
    pca_components = f"PCA ({n_components} components)" if n_components else "none"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                # Encode nominal features with OHE
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("debug_nominal", DebugTransformer(name="Nominal Debug"))
                ]), nominal_features),
                # Encode ordinal&numerical features with RTE
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=50)),
                    # ("debug_numerical", DebugTransformer(name="Numerical Debug"))
                    # ("embedding", RandomTreesEmbedding(random_state=42))
                ]), list(set(X.columns.values) - set(nominal_features))),
            ])),
            # pca_step,
            ("embedding", RandomTreesEmbedding(random_state=42)),
            ("debug_final", DebugTransformer(name="Final Feature Set")),
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
        ]),
        param_grid={
            "embedding__n_estimators": [10, 100],
            "embedding__max_depth": [2, 5],
            "classifier__C": [2, 10]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3, random_state=42)
    )

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f"Log Reg rte test fitting... ")

        # Fit the model for each fold
        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_per_fold.append({
            "Fold": len(metrics_per_fold),
            "AUC": roc_auc_score(y_test, y_test_pred_proba),
            "AP": average_precision_score(y_test, y_test_pred_proba),
            "Sensitivity": recall_score(y_test, y_test_pred, pos_label=1),
            "Specificity": specificity,
            "Precision": precision_score(y_test, y_test_pred, zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average='macro'),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred)
        })

    # Train the final model on the full dataset
    print(f"Log reg rte train fitting... ")
    search.fit(X, y)
    y_train_pred = search.predict(X)
    y_train_pred_proba = search.predict_proba(X)[:, 1]

    # Training metrics
    train_metrics = {
        "AUC": roc_auc_score(y, y_train_pred_proba),
        "AP": average_precision_score(y, y_train_pred_proba),
        "Sensitivity": recall_score(y, y_train_pred, pos_label=1),
        "Specificity": confusion_matrix(y, y_train_pred).ravel()[0] / (
                confusion_matrix(y, y_train_pred).ravel()[0] + confusion_matrix(y, y_train_pred).ravel()[1]),
        "Precision": precision_score(y, y_train_pred, zero_division=0),
        "F1": f1_score(y, y_train_pred, average='macro'),
        "Balanced Accuracy": balanced_accuracy_score(y, y_train_pred)
    }
    best_params = f"{search.best_params_}"
    print(f"Embedding size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, best_params, train_metrics, metrics_per_fold


# n_components aus dem Datensatz nehmen (40 für Posttrauma (shape[1])
def lr_txt_emb(dataset_name, emb_method, feature_extractor, raw_text_summaries, y,
               n_components, n_repeats, max_iter, n_splits=3):
    dataset = dataset_name
    ml_method = "logistic regression"
    concatenation = "no"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)
    is_sentence_transformer = False
    if "gtr-t5-base" in emb_method.lower() or "sentence-t5-base" in emb_method.lower() or "modernbert_embed" in emb_method.lower():
        is_sentence_transformer = True

    pca_components = f"PCA ({n_components} components)" if n_components else "none"

    pipeline_steps = [
        ("aggregator", EmbeddingAggregator(feature_extractor=feature_extractor,
                                           is_sentence_transformer=is_sentence_transformer
                                           ))
    ]
    if n_components:
        pipeline_steps.append(("numerical_scaler", StandardScaler()))
        pipeline_steps.append(("pca", PCA(n_components=n_components)))
        # pipeline_steps.append(("numerical_scaler", MinMaxScaler()))
    else:
        pipeline_steps.append(("numerical_scaler", MinMaxScaler()))

    pipeline_steps.append(("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=max_iter)))

    search = GridSearchCV(
        estimator=Pipeline(pipeline_steps),
        param_grid={
            "classifier__C": [2, 10, 50, 250],
            "aggregator__method": [
                "embedding_cls",
                "embedding_mean_with_cls_and_sep",
                "embedding_mean_without_cls_and_sep"
            ]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=n_repeats, random_state=42)
    )

    for train_index, test_index in skf.split(raw_text_summaries, y):
        print(f"train, text index: {train_index}, {test_index}")
        X_train, X_test = [raw_text_summaries[i] for i in train_index], [raw_text_summaries[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]

        n_samples = len(X_train)
        n_features = X_train[0].shape[0] if hasattr(X_train[0], 'shape') else len(X_train[0])

        # Zeige die Dimensionen an
        print(f"Number of samples (train) (n_samples): {n_samples}")  #
        print(f"Number of samples (test) (n_samples): {len(X_test)}")  #
        print(f"Number of features (n_features): {n_features}")
        print(f"Minimum of samples and features: {min(n_samples, n_features)}")

        # Fit and evaluate
        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        best_param = f"Best params for this fold: {search.best_params_}"
        print(best_param)

        metrics_per_fold.append({
            "Fold": len(metrics_per_fold),
            "AUC": roc_auc_score(y_test, y_test_pred_proba),
            "AP": average_precision_score(y_test, y_test_pred_proba),
            "Sensitivity": recall_score(y_test, y_test_pred, pos_label=1),
            "Specificity": specificity,
            "Precision": precision_score(y_test, y_test_pred, zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average='macro'),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred)
        })

    search.fit(
        raw_text_summaries,
        y
    )

    y_train_pred = search.predict(raw_text_summaries)
    y_train_pred_proba = search.predict_proba(raw_text_summaries)[:, 1]

    # Training metrics
    train_metrics = {
        "AUC": roc_auc_score(y, y_train_pred_proba),
        "AP": average_precision_score(y, y_train_pred_proba),
        "Sensitivity": recall_score(y, y_train_pred, pos_label=1),
        "Specificity": confusion_matrix(y, y_train_pred).ravel()[0] / (
                confusion_matrix(y, y_train_pred).ravel()[0] + confusion_matrix(y, y_train_pred).ravel()[1]),
        "Precision": precision_score(y, y_train_pred, zero_division=0),
        "F1": f1_score(y, y_train_pred, average='macro'),
        "Balanced Accuracy": balanced_accuracy_score(y, y_train_pred)
    }

    best_params = f"{search.best_params_}"

    print(f"embedding size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, best_params, pca_components, train_metrics, metrics_per_fold


def hgbc(dataset_name, X, y, nominal_features, n_repeats=10, n_splits=3):
    dataset = dataset_name
    ml_method = "HistGradientBoosting"
    emb_method = "none"
    concatenation = "no"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)

    search = GridSearchCV(
        estimator=Pipeline([
            ("hist_gb", HistGradientBoostingClassifier(categorical_features=nominal_features))
        ]),
        param_grid={"hist_gb__min_samples_leaf": [5, 10, 15, 20]},
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=n_repeats, random_state=42)
    )

    # Calculate metrics for each fold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f"HGBC test fitting... ")

        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_per_fold.append({
            "Fold": len(metrics_per_fold),
            "AUC": roc_auc_score(y_test, y_test_pred_proba),
            "AP": average_precision_score(y_test, y_test_pred_proba),
            "Sensitivity": recall_score(y_test, y_test_pred, pos_label=1),
            "Specificity": specificity,
            "Precision": precision_score(y_test, y_test_pred, zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average='macro'),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred)
        })

    print(f"HGBC train fitting... ")
    search.fit(X, y)
    y_train_pred = search.predict(X)
    y_train_pred_proba = search.predict_proba(X)[:, 1]

    train_metrics = {
        "AUC": roc_auc_score(y, y_train_pred_proba),
        "AP": average_precision_score(y, y_train_pred_proba),
        "Sensitivity": recall_score(y, y_train_pred, pos_label=1),
        "Specificity": confusion_matrix(y, y_train_pred).ravel()[0] / (
                confusion_matrix(y, y_train_pred).ravel()[0] + confusion_matrix(y, y_train_pred).ravel()[1]),
        "Precision": precision_score(y, y_train_pred, zero_division=0),
        "F1": f1_score(y, y_train_pred, average='macro'),
        "Balanced Accuracy": balanced_accuracy_score(y, y_train_pred)
    }
    best_params = f"{search.best_params_}"

    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, best_params, train_metrics, metrics_per_fold


def hgbc_rte(dataset_name, X, y, nominal_features, n_splits=3):
    ml_method = "HGBC"
    emb_method = "RTE"
    conc = "no"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), nominal_features),
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=30))
                ]), list(set(X.columns.values) - set(nominal_features))),
            ])),
            ("embedding", RandomTreesEmbedding(sparse_output=False, random_state=42)),
            ("hist_gb", HistGradientBoostingClassifier())
        ]),
        param_grid={
            "embedding__n_estimators": [10, 100],
            "embedding__max_depth": [2, 5],
            "hist_gb__min_samples_leaf": [5, 10, 15, 20]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3)
    )

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f"X_train size before: {X_train.shape}")
        print(f"y_train size: {len(y_train)}")
        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_per_fold.append({
            "Fold": len(metrics_per_fold),
            "AUC": roc_auc_score(y_test, y_test_pred_proba),
            "AP": average_precision_score(y_test, y_test_pred_proba),
            "Sensitivity": recall_score(y_test, y_test_pred, pos_label=1),
            "Specificity": specificity,
            "Precision": precision_score(y_test, y_test_pred, zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average='macro'),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred)
        })

    search.fit(X, y)
    y_train_pred = search.predict(X)
    y_train_pred_proba = search.predict_proba(X)[:, 1]

    train_metrics = {
        "AUC": roc_auc_score(y, y_train_pred_proba),
        "AP": average_precision_score(y, y_train_pred_proba),
        "Sensitivity": recall_score(y, y_train_pred, pos_label=1),
        "Specificity": confusion_matrix(y, y_train_pred).ravel()[0] / (
                confusion_matrix(y, y_train_pred).ravel()[0] + confusion_matrix(y, y_train_pred).ravel()[1]),
        "Precision": precision_score(y, y_train_pred, zero_division=0),
        "F1": f1_score(y, y_train_pred, average='macro'),
        "Balanced Accuracy": balanced_accuracy_score(y, y_train_pred)
    }
    best_params = f"{search.best_params_}"

    hgbc_rt_emb_train_score = roc_auc_score(y, search.predict_proba(X)[:, 1])
    print(f"best hyperparameters: {best_params}")
    print(f"lr_ran_tree_emb_train_score: {hgbc_rt_emb_train_score}")

    return dataset_name, ml_method, emb_method, conc, best_params, train_metrics, metrics_per_fold


def hgbc_txt_emb(dataset_name, emb_method, feature_extractor, summaries, y,
                 n_components, n_repeats, n_splits=3):
    print(f"Started: hgbc_txt_emb with {feature_extractor}")
    dataset = dataset_name
    ml_method = "HistGradientBoosting"
    emb_method = emb_method
    concatenation = "no"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)
    is_sentence_transformer = False
    if "gtr-t5-base" in emb_method.lower() or "sentence-t5-base" in emb_method.lower() or "modernbert_embed" in emb_method.lower():
        is_sentence_transformer = True

    pca_components = f"PCA ({n_components} components)" if n_components else "none"

    pipeline_steps = [
        ("aggregator", EmbeddingAggregator(
            feature_extractor=feature_extractor,
            is_sentence_transformer=is_sentence_transformer))
    ]
    if n_components:
        pipeline_steps.append(("numerical_scaler", StandardScaler()))
        pipeline_steps.append(("pca", PCA(n_components=n_components)))

    pipeline_steps.append(("hist_gb", HistGradientBoostingClassifier()))

    search = GridSearchCV(
        estimator=Pipeline(pipeline_steps),
        param_grid={
            "hist_gb__min_samples_leaf": [5, 10, 15, 20],
            "aggregator__method": ["embedding_cls",
                                   "embedding_mean_with_cls_and_sep",
                                   "embedding_mean_without_cls_and_sep"]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=n_repeats)
    )
    print(f"len of summaries: {len(summaries)}")
    print(f"len of y: {len(y)}")

    for train_index, test_index in skf.split(summaries, y):
        X_train, X_test = [summaries[i] for i in train_index], [summaries[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_test = np.array(X_train), np.array(X_test)

        search.fit(
            np.array(X_train),
            y_train
        )
        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_per_fold.append({
            "Fold": len(metrics_per_fold),
            "AUC": roc_auc_score(y_test, y_test_pred_proba),
            "AP": average_precision_score(y_test, y_test_pred_proba),
            "Sensitivity": recall_score(y_test, y_test_pred, pos_label=1),
            "Specificity": specificity,
            "Precision": precision_score(y_test, y_test_pred, zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average='macro'),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred)
        })

    # train on the full dataset
    search.fit(
        np.array(summaries),
        y
    )

    y_train_pred = search.predict(np.array(summaries))
    y_train_pred_proba = search.predict_proba(np.array(summaries))[:, 1]

    # Training metrics
    train_metrics = {
        "AUC": roc_auc_score(y, y_train_pred_proba),
        "AP": average_precision_score(y, y_train_pred_proba),
        "Sensitivity": recall_score(y, y_train_pred, pos_label=1),
        "Specificity": confusion_matrix(y, y_train_pred).ravel()[0] / (
                confusion_matrix(y, y_train_pred).ravel()[0] + confusion_matrix(y, y_train_pred).ravel()[1]),
        "Precision": precision_score(y, y_train_pred, zero_division=0),
        "F1": f1_score(y, y_train_pred, average='macro'),
        "Balanced Accuracy": balanced_accuracy_score(y, y_train_pred)
    }
    aggregator = search.best_estimator_.named_steps['aggregator']

    try:
        # Print the selected method
        print(f"best aggregator method: {search.best_params_['aggregator__method']}")
        if hasattr(aggregator, 'aggregation_info'):
            print(f"Aggregator info: {aggregator.aggregation_info}")
        else:
            print("Aggregator does not expose additional info.")
    except Exception as e:
        print(f"Could not retrieve aggregator details: {e}")

    best_params = f"{search.best_params_}"

    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, best_params, pca_components, train_metrics, metrics_per_fold


# läuft
def concat_lr_txt_emb(dataset_name, emb_method,
                      feature_extractor, raw_text_summaries,
                      X_tabular, y, nominal_features, text_feature_column_name,
                      imp_max_iter, class_max_iter, n_repeats,
                      n_components, n_splits=3):
    start_time = time.time()
    readable_time = time.strftime("%H:%M:%S", time.localtime(start_time))
    print(f"Starting the concat_lr_txt_emb method {readable_time}")

    dataset = dataset_name
    ml_method = "Logistic Regression"
    emb_method = emb_method
    concatenation = "yes"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)

    # add new column (text summaries)
    text_features = [text_feature_column_name]
    X_tabular[text_feature_column_name] = raw_text_summaries

    # define numerical features
    numerical_features = list(set(X_tabular.columns) -
                              set(nominal_features) -
                              set(text_features))
    print(f"Len numerical features: {len(numerical_features)}")  # muss 41X82
    print(f"Numerical features identified: {numerical_features}")
    print(f"Setting up the pipeline at {time.strftime('%H:%M:%S', time.localtime(time.time()))}")
    print(f"Tabelle Größe {X_tabular.shape}")  # muss 41X82
    print(f"All columns: {X_tabular.columns}")

    pca_components = f"PCA ({n_components} components)" \
        if n_components else "none"

    is_sentence_transformer = False
    if "gtr_t5_base" in emb_method.lower() or "sentence_t5_base" in emb_method.lower() or "modernbert_embed" in emb_method.lower():
        is_sentence_transformer = True

    pipeline_text_steps = [
        ("embedding_aggregator", EmbeddingAggregator(
            feature_extractor=feature_extractor,
            is_sentence_transformer=is_sentence_transformer
        )),
    ]
    if n_components:
        pipeline_text_steps.append(("numerical_scaler", StandardScaler()))
        pipeline_text_steps.append(("pca", PCA(n_components=n_components)))
    else:
        pipeline_text_steps.append(("numerical_scaler", MinMaxScaler()))

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
                    ("numerical_scaler", MinMaxScaler())
                ]), numerical_features),
                ("text", Pipeline(pipeline_text_steps), text_features),
            ])),
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=class_max_iter))
        ]),
        param_grid={
            "classifier__C": [2, 10, 50, 250],
            "transformer__text__embedding_aggregator__method": [
                "embedding_cls",
                "embedding_mean_with_cls_and_sep",
                "embedding_mean_without_cls_and_sep"
            ]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=n_repeats)
    )

    for train_index, test_index in skf.split(X_tabular, y):
        X_train, X_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_per_fold.append({
            "Fold": len(metrics_per_fold),
            "AUC": roc_auc_score(y_test, y_test_pred_proba),
            "AP": average_precision_score(y_test, y_test_pred_proba),
            "Sensitivity": recall_score(y_test, y_test_pred, pos_label=1),
            "Specificity": specificity,
            "Precision": precision_score(y_test, y_test_pred, zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average='macro'),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred)
        })

    search.fit(X_tabular, y)

    y_train_pred = search.predict(X_tabular)
    y_train_pred_proba = search.predict_proba(X_tabular)[:, 1]

    print(f"Shape X_tabular: {X_tabular.shape}")
    print(f"y shape: {y.shape}")  # Should be (82,)
    print(f"y_train_pred shape: {y_train_pred.shape}")  # Should also be (82,)

    train_metrics = {
        "AUC": roc_auc_score(y, y_train_pred_proba),
        "AP": average_precision_score(y, y_train_pred_proba),
        "Sensitivity": recall_score(y, y_train_pred, pos_label=1),
        "Specificity": confusion_matrix(y, y_train_pred).ravel()[0] / (
                confusion_matrix(y, y_train_pred).ravel()[0] + confusion_matrix(y, y_train_pred).ravel()[1]),
        "Precision": precision_score(y, y_train_pred, zero_division=0),
        "F1": f1_score(y, y_train_pred, average='macro'),
        "Balanced Accuracy": balanced_accuracy_score(y, y_train_pred)
    }

    finish_time = time.time()
    readable_time = time.strftime("%H:%M:%S", time.localtime(finish_time))
    print(f"Finished the concat_lr_txt_emb method {readable_time}")

    best_params = f"{search.best_params_}"

    print(f"Combined feature size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, best_params, pca_components, train_metrics, metrics_per_fold


def concat_lr_rte(dataset_name, X_tabular,
                  nominal_features, y, n_repeats,
                  imp_max_iter, class_max_iter, pca_n_comp,
                  n_splits=3):
    dataset = dataset_name
    ml_method = "Logistic Regression"
    emb_method = "Random Trees Embedding"
    concatenation = "yes"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)
    pca_transformer = PCA(n_components=pca_n_comp, svd_solver='auto') if pca_n_comp is not None else "passthrough"
    numerical_features = list(set(X_tabular.columns) - set(nominal_features))

    num_pipeline_steps = [
        ("debug_numerical", DebugTransformer(name="Numerical Debug")),
        ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
        ("debug_numerical_after", DebugTransformer(name="Numerical Debug after"))
    ]
    if pca_n_comp:
        num_pipeline_steps.append(("scaler", StandardScaler()))

    pipeline = Pipeline([
        ("feature_combiner", FeatureUnion([
            # Verarbeitung der tabellarischen Daten
            ("raw", ColumnTransformer([
                ("nominal", Pipeline([
                    ("debug_nominal", DebugTransformer(name="Nominal Debug")),  # 5
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore")),  # 14
                    ("debug_nominal_after", DebugTransformer(name="Nominal Debug after"))
                ]), nominal_features),
                ("numerical", Pipeline([
                    ("debug_numerical", DebugTransformer(name="Numerical Debug")),
                    ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),  # 35
                    ("debug_numerical_after", DebugTransformer(name="Numerical Debug after"))
                ]), numerical_features),  # 49
            ], remainder="passthrough")),
            # Verarbeitung der RT Embeddings
            ("embeddings", Pipeline([
                ("transformer", ColumnTransformer([
                    ("nominal", Pipeline([
                        ("debug_nominal_emb", DebugTransformer(name="Nominal Debug Emb after")),
                        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                        # ("nominal_encoder", OneHotEncoder(handle_unknown="ignore")),
                        ("nominal_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                        ("debug_nominal_emb_after", DebugTransformer(name="Nominal Debug Emb"))
                    ]), nominal_features),
                    ("numerical", Pipeline(
                        steps=num_pipeline_steps
                    ), numerical_features),
                ], remainder="passthrough")),
                ("debug_embedding", DebugTransformer(name="Embedding Debug")),
                ("embedding", RandomTreesEmbedding(random_state=42)),  # check
                ("pca", pca_transformer),
                ("debug_embedding_after", DebugTransformer(name="Embedding Debug after"))
            ]))
        ])),
        ("debug_final", DebugTransformer(name="Final Feature Set")),
        ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=class_max_iter))
    ])

    param_grid = {
        "classifier__C": [2, 10],
        "feature_combiner__embeddings__embedding__n_estimators": [10, 100],
        "feature_combiner__embeddings__embedding__max_depth": [2, 5],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=n_repeats)
    )

    for train_index, test_index in skf.split(X_tabular, y):
        X_tab_train, X_tab_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        search.fit(X_tab_train, y_train)

        y_test_pred = search.predict(X_tab_test)
        y_test_pred_proba = search.predict_proba(X_tab_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_per_fold.append({
            "Fold": len(metrics_per_fold),
            "AUC": roc_auc_score(y_test, y_test_pred_proba),
            "AP": average_precision_score(y_test, y_test_pred_proba),
            "Sensitivity": recall_score(y_test, y_test_pred, pos_label=1),
            "Specificity": specificity,
            "Precision": precision_score(y_test, y_test_pred, zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average='macro'),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred)
        })

    search.fit(X_tabular, y)
    y_train_pred = search.predict(X_tabular)
    y_train_pred_proba = search.predict_proba(X_tabular)[:, 1]

    train_metrics = {
        "AUC": roc_auc_score(y, y_train_pred_proba),
        "AP": average_precision_score(y, y_train_pred_proba),
        "Sensitivity": recall_score(y, y_train_pred, pos_label=1),
        "Specificity": confusion_matrix(y, y_train_pred).ravel()[0] / (
                confusion_matrix(y, y_train_pred).ravel()[0] + confusion_matrix(y, y_train_pred).ravel()[1]),
        "Precision": precision_score(y, y_train_pred, zero_division=0),
        "F1": f1_score(y, y_train_pred, average='macro'),
        "Balanced Accuracy": balanced_accuracy_score(y, y_train_pred)
    }

    best_params = f"{search.best_params_}"

    print(f"Feature set size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, best_params, pca_n_comp, train_metrics, metrics_per_fold


# läuft
def concat_txt_hgbc(dataset_name, emb_method,
                    X_tabular, y, text_feature_column_name,
                    nominal_features, feature_extractor,
                    raw_text_summaries, n_repeats,
                    n_components, n_splits=3):
    dataset = dataset_name
    ml_method = "HistGradientBoostingClassifier"
    emb_method = emb_method
    concatenation = "yes"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)

    # add text as a new column
    text_features = [text_feature_column_name]
    X_tabular[text_feature_column_name] = raw_text_summaries

    # separate non-text features
    non_text_columns = list(set(X_tabular.columns) -
                            set(text_features))

    print(f"All columns length: {X_tabular.shape}")
    print(f"Non-text columns length: {len(X_tabular[non_text_columns])}")
    print(f"Non-text columns shape: {X_tabular[non_text_columns].shape}")

    pca_components = f"PCA ({n_components} components)" \
        if n_components else "none"

    is_sentence_transformer = False
    if "gtr-t5-base" in emb_method.lower() or "sentence-t5-base" in emb_method.lower() or "modernbert_embed" in emb_method.lower():
        is_sentence_transformer = True

    pipeline_text_steps = [
        ("embedding_aggregator", EmbeddingAggregator(
            feature_extractor=feature_extractor,
            is_sentence_transformer=is_sentence_transformer)),
        ("debug_text", DebugTransformer(name="Text Debug"))
    ]
    if n_components:
        pipeline_text_steps.append(("numerical_scaler", StandardScaler()))
        pipeline_text_steps.append(("pca", PCA(n_components=n_components)))
    else:
        pipeline_text_steps.append(("numerical_scaler", MinMaxScaler()))

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("text", Pipeline(pipeline_text_steps), text_features)
            ])),
            ("classifier", HistGradientBoostingClassifier())
        ]),
        param_grid={
            "classifier__min_samples_leaf": [5, 10, 15, 20],
            "transformer__text__embedding_aggregator__method": [
                "embedding_cls",
                "embedding_mean_with_cls_and_sep",
                "embedding_mean_without_cls_and_sep"
            ]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=n_repeats)
    )

    for train_index, test_index in skf.split(X_tabular, y):
        X_train, X_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f"Length of X_tab_train: {len(X_train)}")
        print(f"Length of y_train: {len(y_train)}")

        assert len(X_train) == len(y_train), "Mismatch in training data sizes"

        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_per_fold.append({
            "Fold": len(metrics_per_fold),
            "AUC": roc_auc_score(y_test, y_test_pred_proba),
            "AP": average_precision_score(y_test, y_test_pred_proba),
            "Sensitivity": recall_score(y_test, y_test_pred, pos_label=1),
            "Specificity": specificity,
            "Precision": precision_score(y_test, y_test_pred, zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average='macro'),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred)
        })

    print(f"X_tabular len: {len(X_tabular)}")
    print(f"Text_features len: {len(text_features)}")  # muss 82 sein
    print(f"y len: {len(y)}")
    assert len(X_tabular) == len(y), "Mismatch in training data sizes"

    search.fit(X_tabular, y)
    y_train_pred = search.predict(X_tabular)
    y_train_pred_proba = search.predict_proba(X_tabular)[:, 1]

    print(f"X_tabular shape {X_tabular.shape}")
    print(f"y shape: {y.shape}")  # Should be (82,)
    print(f"y_train_pred shape: {y_train_pred.shape}")  # Should also be (82,)

    train_metrics = {
        "AUC": roc_auc_score(y, y_train_pred_proba),
        "AP": average_precision_score(y, y_train_pred_proba),
        "Sensitivity": recall_score(y, y_train_pred, pos_label=1),
        "Specificity": confusion_matrix(y, y_train_pred).ravel()[0] / (
                confusion_matrix(y, y_train_pred).ravel()[0] +
                confusion_matrix(y, y_train_pred).ravel()[1]),
        "Precision": precision_score(y, y_train_pred, zero_division=0),
        "F1": f1_score(y, y_train_pred, average='macro'),
        "Balanced Accuracy": balanced_accuracy_score(y, y_train_pred)
    }

    best_params = f"{search.best_params_}"
    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return (dataset, ml_method, emb_method, concatenation, best_params,
            pca_components, train_metrics, metrics_per_fold)


def concat_hgbc_rte(dataset_name, X_tabular, y, nominal_features, n_repeats,
                    pca_n_comp, imp_max_iter, n_splits=3):
    dataset = dataset_name
    ml_method = "HistGradientBoosting"
    emb_method = "Random Trees Embedding"
    concatenation = "yes"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)
    categorical_indices = [X_tabular.columns.get_loc(col) for col in nominal_features]
    numerical_features = list(set(X_tabular.columns) - set(nominal_features))
    pca_transformer = PCA(n_components=pca_n_comp, svd_solver='auto') if pca_n_comp is not None else "passthrough"

    print(f"type of X: {type(X_tabular)}")
    num_pipeline_steps = [
        ("debug_numerical", DebugTransformer(name="Numerical Debug")),
        ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
        ("debug_numerical_after", DebugTransformer(name="Numerical Debug after"))
    ]
    if pca_n_comp:
        num_pipeline_steps.append(("scaler", StandardScaler()))

    pipeline = Pipeline([
        ("feature_combiner", FeatureUnion([
            ("raw", "passthrough"),
            ("embeddings", Pipeline([
                ("transformer", ColumnTransformer([
                    ("nominal", Pipeline([
                        ("debug_nominal", DebugTransformer(name="Nominal Debug")),
                        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                        ("nominal_encoder", OneHotEncoder(handle_unknown="ignore")),
                        ("debug_nominal_after", DebugTransformer(name="Nominal Debug after"))
                    ]), nominal_features),
                    ("numerical", Pipeline(
                        steps=num_pipeline_steps
                    ), numerical_features)
                ])),
                ("debug_embedding", DebugTransformer(name="Embedding Debug")),
                ("embedding", RandomTreesEmbedding(sparse_output=False, random_state=42)),
                ("pca", pca_transformer),
                ("debug_embedding_after", DebugTransformer(name="Embedding Debug after"))
            ]))
        ])),
        ("debug_final", DebugTransformer(name="Final Feature Set")),
        ("hist_gb", HistGradientBoostingClassifier(random_state=42, categorical_features=categorical_indices))
    ])

    param_grid = {
        "hist_gb__min_samples_leaf": [5, 10, 15, 20],
        "feature_combiner__embeddings__embedding__n_estimators": [10, 100],  # decreased for small data
        "feature_combiner__embeddings__embedding__max_depth": [2, 5]  # decreased for small data
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=n_repeats)
    )

    for train_index, test_index in skf.split(X_tabular, y):
        X_train, X_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        search.fit(X_train, y_train)
        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_per_fold.append({
            "Fold": len(metrics_per_fold),
            "AUC": roc_auc_score(y_test, y_test_pred_proba),
            "AP": average_precision_score(y_test, y_test_pred_proba),
            "Sensitivity": recall_score(y_test, y_test_pred, pos_label=1),
            "Specificity": specificity,
            "Precision": precision_score(y_test, y_test_pred, zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average='macro'),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred)
        })

    search.fit(X_tabular, y)
    y_pred = search.predict(X_tabular)
    y_train_pred_proba = search.predict_proba(X_tabular)[:, 1]

    train_metrics = {
        "AUC": roc_auc_score(y, y_train_pred_proba),
        "AP": average_precision_score(y, y_train_pred_proba),
        "Sensitivity": recall_score(y, y_pred, pos_label=1),
        "Specificity": confusion_matrix(y, y_pred).ravel()[0] / (
                confusion_matrix(y, y_pred).ravel()[0] +
                confusion_matrix(y, y_pred).ravel()[1]),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, average='macro'),
        "Balanced Accuracy": balanced_accuracy_score(y, y_pred)
    }

    best_params = f"{search.best_params_}"

    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Combined train score: {train_metrics}")
    print(f"Combined test scores: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, best_params, pca_n_comp, train_metrics, metrics_per_fold


class DebugTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, name="Step"):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"{self.name}: Input shape {X.shape}")
        if isinstance(X, pd.DataFrame):
            X_transformed = X.to_numpy()
        else:
            X_transformed = X
        print(f"{self.name}: Output shape {X_transformed.shape}")
        return X_transformed
