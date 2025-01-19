import os
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding, HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler#, OrdinalEncoder
#from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix,
    average_precision_score
)

from csv_parser import create_patient_summaries
from text_emb_aggregator import EmbeddingAggregator


# Load features
def load_features(file_path="X.csv", delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter)
    print(f"features: {data}")
    return data


# Load labels
def load_labels(file_path="y.csv", delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter)
    return np.array(data.values.ravel())


# Load features as text summaries (create if doesn't exist)
def load_summaries():
    if not os.path.exists("Summaries.txt"):
        create_patient_summaries()
    with open("Summaries.txt", "r") as file:
        summaries_list = [line.strip() for line in file.readlines()]
    return summaries_list


def logistic_regression(dataset_name, X, y, nominal_features, n_splits=3, n_components=None):
    # Todo: try encoding categ. features with OHE (after finding all categ. features (with Ricardo))
    # for csv format
    # dataset und fold(index von test scores) in der anderen Datei
    dataset = dataset_name
    ml_method = "logistic regression"
    emb_method = "none"
    pca_components = f"PCA ({n_components} components)" if n_components else "none"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

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
                    ("numerical_imputer", IterativeImputer(max_iter=30)),
                    ("numerical_scaler", MinMaxScaler())
                ] + ([pca_step] if pca_step else [])), numerical_features),
            ])),
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
        ]),
        param_grid={"classifier__C": [2, 10, 50, 250]},
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3)
    )

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

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

    print(f"Feature set size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, pca_components, train_metrics, metrics_per_fold


def lr_rt_emb(dataset_name, X, y, nominal_features, n_splits=3, n_components=None):
    dataset = dataset_name
    ml_method = "logistic regression"
    emb_method = "random tree embedding"
    concatenation = "no"
    pca_components = f"PCA ({n_components} components)" if n_components else "none"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

    pca_step = ("pca", PCA(n_components=n_components)) if n_components else None

    # Todo Question: Why both RTE and OHE on categ. features??
    # Todo: Try OHE on categorical + RTE on numerical (better for Log.Reg. ?)
    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                # Encode nominal features with OHE
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), nominal_features),
                # Encode ordinal&numerical features with RTE
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=50)),
                    ("embedding", RandomTreesEmbedding())
                ]), list(set(X.columns.values) - set(nominal_features))),
            ])),
            # pca_step,
            # ("embedding", RandomTreesEmbedding()),
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
        ]),
        param_grid={
            #"embedding__n_estimators": [10, 100, 1000],
            #"embedding__max_depth": [2, 5, 10, 15],
            "transformer__numerical__embedding__n_estimators": [10, 100, 1000],  # Adjusted for nesting
            "transformer__numerical__embedding__max_depth": [2, 5, 10, 15],
            "classifier__C": [2, 10, 50, 250]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3)
    )

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

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

    print(f"Embedding size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, train_metrics, metrics_per_fold


def lr_txt_emb(dataset_name, emb_method, feature_extractor, summaries, y, n_splits=3, n_components=None):
    # Todo! Wird die beste Aggregierung ausgegeben?
    dataset = dataset_name
    ml_method = "logistic regression"
    concatenation = "no"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

    search = GridSearchCV(
        estimator=Pipeline([
            ("aggregator", EmbeddingAggregator(feature_extractor)),
            ("numerical_scaler", MinMaxScaler()), # test vs. StandardScaler
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
        ]),
        param_grid={
            "classifier__C": [2, 10, 50, 250],
            "aggregator__method": [
                "embedding_cls",
                "embedding_mean_with_cls_and_sep",
                "embedding_mean_without_cls_and_sep"
            ]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3)
    )
    for train_index, test_index in skf.split(summaries, y):
        X_train, X_test = [summaries[i] for i in train_index], [summaries[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit and evaluate
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

    search.fit(
        summaries,
        y
    )

    y_train_pred = search.predict(summaries)
    y_train_pred_proba = search.predict_proba(summaries)[:, 1]

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
    print(f"embedding size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, train_metrics, metrics_per_fold


def hgbc(dataset_name, X, y, nominal_features, n_splits=3):
    dataset = dataset_name
    ml_method = "HistGradientBoosting"
    emb_method = "none"
    concatenation = "no"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

    # Define GridSearchCV once before the loop
    search = GridSearchCV(
        estimator=Pipeline([
            ("hist_gb", HistGradientBoostingClassifier(categorical_features=nominal_features))
        ]),
        param_grid={"hist_gb__min_samples_leaf": [5, 10, 15, 20]},
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3)
    )

    # Calculate metrics for each fold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

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

    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, train_metrics, metrics_per_fold


def hgbc_ran_tree_emb(dataset_name, X, y, nominal_features, n_splits=3):
    dataset = dataset_name
    ml_method = "HistGradientBoosting"
    emb_method = "random tree embedding"
    concatenation = "no"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

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
            ("embedding", RandomTreesEmbedding()),
            ("hist_gb", HistGradientBoostingClassifier())
        ]),
        param_grid={
            "embedding__n_estimators": [10, 100, 1000],
            "embedding__max_depth": [2, 5, 10, 15],
            "hist_gb__min_samples_leaf": [5, 10, 15, 20]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3)
    )

    # Metrics for each fold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

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

    # Train on the full dataset
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

    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, train_metrics, metrics_per_fold


def hgbc_txt_emb(dataset_name, emb_method, feature_extractor, summaries, y, n_splits=3): # Todo! Wird die beste Aggregierung ausgegeben?
    dataset = dataset_name
    ml_method = "HistGradientBoosting"
    emb_method = emb_method
    concatenation = "no"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

    search = GridSearchCV(
        estimator=Pipeline([
            ("aggregator", EmbeddingAggregator(feature_extractor)),
            ("numerical_scaler", MinMaxScaler()),
            ("hist_gb", HistGradientBoostingClassifier()),
        ]),
        param_grid={
            "hist_gb__min_samples_leaf": [5, 10, 15, 20],
            "aggregator__method": ["embedding_cls", "embedding_mean_with_cls_and_sep",
                                   "embedding_mean_without_cls_and_sep"]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3)
        # todo: macht es Unterschied? Bzw. wird jetzt die beste Aggrerg. ausgewält
    )

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

        # Include any additional details about the aggregator, if available
        if hasattr(aggregator, 'aggregation_info'):
            print(f"Aggregator info: {aggregator.aggregation_info}")
        else:
            print("Aggregator does not expose additional info.")
    except Exception as e:
        print(f"Could not retrieve aggregator details: {e}")

    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, train_metrics, metrics_per_fold


def concat_lr_txt_emb(dataset_name, emb_method, X_tabular, summaries, feature_extractor, nominal_features, y,
                      n_splits=3):
    start_time = time.time()
    print(f"Starting the concat_lr_txt_emb method {start_time}")
    dataset = dataset_name
    ml_method = "Logistic Regression"
    emb_method = emb_method
    concatenation = "yes"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

    numerical_features = list(set(X_tabular.columns) - set(nominal_features))
    print(f"Numerical features identified: {numerical_features}")

    print(f"Setting up the pipeline at {time.time()}")

    tabular_pipeline = Pipeline([
        ("tabular_transformer", ColumnTransformer([
            ("nominal", Pipeline([
                ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), nominal_features),
            ("numerical", Pipeline([
                ("numerical_imputer", IterativeImputer(max_iter=50)),
                ("numerical_scaler", MinMaxScaler())
            ]), numerical_features)
        ]))
    ])

    embeddings_pipeline = Pipeline([
        ("embedding_transformer", EmbeddingAggregator(feature_extractor)),
        ("scaler", MinMaxScaler())
    ])

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("tabular", tabular_pipeline),
            ("embeddings", embeddings_pipeline)
        ])),
        ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
    ])

    """pipeline = Pipeline([
        ("feature_combiner", ColumnTransformer([
            # Verarbeitung der tabellarischen Daten
            ("tabular", ColumnTransformer([
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),  # try other?
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), nominal_features),
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=50)),
                    ("numerical_scaler", MinMaxScaler())
                ]), numerical_features),
            ]), X_tabular.columns),

            # Verarbeitung der Embeddings
            ("embeddings", Pipeline([
                ("aggregator", EmbeddingAggregator(feature_extractor)),
                ("numerical_scaler", MinMaxScaler())
            ]), summaries)
        ])),
        ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
    ])"""
    print("Setting up the parameter grid...")
    param_grid = {
        "classifier__C": [2, 10, 50, 250],
        "embedding_transformer__embeddings__aggregator__method": [
            "embedding_cls",
            "embedding_mean_with_cls_and_sep",
            "embedding_mean_without_cls_and_sep"
        ]
    }

    print("Initializing GridSearchCV...")
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3)
    )

    print("Starting cross-validation...")
    # Cross-Validation
    for fold, (train_index, test_index) in enumerate(skf.split(X_tabular, y)):
        print(f"Processing fold {fold + 1}...")
        # Aufteilen der tabellarischen Daten und Embeddings
        X_tab_train, X_tab_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        summaries_train = [summaries[i] for i in train_index]
        summaries_test = [summaries[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]

        summaries_train_array = np.array(summaries_train)
        if summaries_train_array.ndim == 1:
            summaries_train_array = summaries_train_array.reshape(-1, 1)

        # Create a DataFrame for embeddings
        summaries_train_df = pd.DataFrame(
            summaries_train_array,
            columns=[f"embedding_{i}" for i in range(summaries_train_array.shape[1])]
        )

        summaries_test_array = np.array(summaries_test)
        if summaries_test_array.ndim == 1:
            summaries_test_array = summaries_test_array.reshape(-1, 1)

        # Create a DataFrame for embeddings
        summaries_test_df = pd.DataFrame(
            summaries_test_array,
            columns=[f"embedding_{i}" for i in range(summaries_test_array.shape[1])]
        )

        print(f"Fitting the model for fold {fold + 1}...")
        print(f"X_tab_train shape: {X_tab_train.shape}")
        print(f"Number of summaries_train: {len(summaries_train)}")
        print(f"y_train shape: {y_train.shape}")

        #combined_train = np.hstack((X_tab_train.values, summaries_train_array))
        #combined_test = np.hstack((X_tab_test.values, summaries_test_array))

        # Combine X_tabular and embeddings
        combined_train = pd.concat([X_tab_train.reset_index(drop=True), summaries_train_df.reset_index(drop=True)],
                                   axis=1)
        combined_test = pd.concat([X_tab_test.reset_index(drop=True), summaries_test_df.reset_index(drop=True)], axis=1)

        #search.fit({"tabular": X_tab_train, "embeddings": summaries_train}, y_train)
        print(f"Combined train columns: {combined_train.columns}")
        print(f"Combined test columns: {combined_test.columns}")

        print(f"Combined train shape: {combined_train.shape}")
        print(f"Combined test shape: {combined_test.shape}")
        search.fit(combined_train, y_train)

        print(f"Making predictions for fold {fold + 1}...")
        #y_test_pred = search.predict({"tabular": X_tab_test, "embeddings": summaries_test})
        #y_test_pred_proba = search.predict_proba({"tabular": X_tab_test, "embeddings": summaries_test})[:, 1]
        y_test_pred = search.predict(combined_test)
        y_test_pred_proba = search.predict(combined_test)[:, 1]

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

    print("Fitting the model on the entire dataset...")

    summaries_array = np.array(summaries)
    if summaries_array.ndim == 1:
        summaries_array = summaries_array.reshape(-1, 1)

    summaries_df = pd.DataFrame(
        summaries_array,
        columns=[f"embedding_{i}" for i in range(summaries_array.shape[1])]
    )

    combined_all = pd.concat([X_tabular.reset_index(drop=True), summaries_df.reset_index(drop=True)], axis=1)
    print(f"Combined all columns: {combined_all.columns}")
    print(f"Combined all shape: {combined_all.shape}")
    search.fit(combined_all, y)

    y_train_pred = search.predict(combined_all)
    y_train_pred_proba = search.predict_proba(combined_all)[:, 1]

    print("Calculating training metrics...")
    # Calculate training metrics
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

    print(f"Feature set size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    print(f"Completed in {time.time() - start_time:.2f} seconds!")
    return dataset, ml_method, emb_method, concatenation, train_metrics, metrics_per_fold


def concat_lr_tab_rt_emb(dataset_name, X_tabular, summaries, nominal_features, y, n_splits=3):
    dataset = dataset_name
    ml_method = "Logistic Regression"
    emb_method = "Random Trees Embedding"
    concatenation = "yes"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

    numerical_features = list(set(X_tabular.columns) - set(nominal_features))

    pipeline = Pipeline([
        ("feature_combiner", ColumnTransformer([
            # Verarbeitung der tabellarischen Daten
            ("tabular", ColumnTransformer([
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), nominal_features),
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=30)),
                    ("numerical_scaler", MinMaxScaler())
                ]), numerical_features),
            ]), X_tabular.columns),

            # Verarbeitung der RT Embeddings
            ("embeddings", Pipeline([
                ("transformer", ColumnTransformer([
                    ("nominal", Pipeline([
                        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                        ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                    ]), nominal_features),
                    ("numerical", Pipeline([
                        ("numerical_imputer", IterativeImputer(max_iter=30))
                    ]), numerical_features),
                ])),
                ("embedding", RandomTreesEmbedding())
            ]), summaries)
        ])),
        ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
    ])

    param_grid = {
        "classifier__C": [2, 10, 50, 250],
        "embedding__n_estimators": [10, 100, 1000],
        "embedding__max_depth": [2, 5, 10, 15],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3)
    )

    # Cross-Validation
    for train_index, test_index in skf.split(X_tabular, y):
        # Aufteilen der tabellarischen Daten und Embeddings
        X_tab_train, X_tab_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        summaries_train = [summaries[i] for i in train_index]
        summaries_test = [summaries[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model
        search.fit({"tabular": X_tab_train, "embeddings": summaries_train}, y_train)

        # Predictions and probabilities
        y_test_pred = search.predict({"tabular": X_tab_test, "embeddings": summaries_test})
        y_test_pred_proba = search.predict_proba({"tabular": X_tab_test, "embeddings": summaries_test})[:, 1]

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

        # Train on the full dataset
    search.fit({"tabular": X_tabular, "embeddings": summaries}, y)
    y_train_pred = search.predict({"tabular": X_tabular, "embeddings": summaries})
    y_train_pred_proba = search.predict_proba({"tabular": X_tabular, "embeddings": summaries})[:, 1]

    # Calculate training metrics
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

    print(f"Feature set size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, train_metrics, metrics_per_fold


# TODO! Anpassen
def concat_txt_tab_hgbc(dataset_name, emb_method, X_tabular, y, nominal_features, feature_extractor, summaries, n_splits=3):
    dataset = dataset_name
    ml_method = "HistGradientBoosting"
    emb_method = emb_method
    concatenation = "yes"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

    embedding_pipeline = Pipeline([
        ("aggregator", EmbeddingAggregator(feature_extractor)),
        ("scaler", MinMaxScaler())
    ])

    param_grid = {
        "hist_gb__min_samples_leaf": [5, 10, 15, 20],
        "embedding__aggregator__method": [
            "embedding_cls",
            "embedding_mean_with_cls_and_sep",
            "embedding_mean_without_cls_and_sep"
        ]
    }

    for train_index, test_index in skf.split(X_tabular, y):
        # Train-Test-Split für tabellarische und Embedding-Daten
        X_tab_train, X_tab_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        summaries_train = [summaries[i] for i in train_index]
        summaries_test = [summaries[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_embeddings_train = embedding_pipeline.fit_transform(summaries_train)
        X_embeddings_test = embedding_pipeline.transform(summaries_test)

        # Kombination von tabellarischen Daten und Embeddings
        X_train_combined = np.hstack([X_tab_train, X_embeddings_train])  # todo: Dimension ausgeben und checken
        X_test_combined = np.hstack([X_tab_test, X_embeddings_test])

        # Modelltraining und Bewertung
        search = GridSearchCV(
            estimator=HistGradientBoostingClassifier(categorical_features=nominal_features),
            param_grid=param_grid,
            scoring="neg_log_loss",
            cv=RepeatedStratifiedKFold(n_splits=3)
        )
        search.fit(X_train_combined, y_train)
        y_pred_proba = search.predict_proba(X_test_combined)[:, 1]

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, search.predict(X_test_combined)).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_per_fold.append({
            "Fold": len(metrics_per_fold),
            "AUC": roc_auc_score(y_test, y_pred_proba),
            "AP": average_precision_score(y_test, y_pred_proba),
            "Sensitivity": recall_score(y_test, search.predict(X_test_combined), pos_label=1),
            "Specificity": specificity,
            "Precision": precision_score(y_test, search.predict(X_test_combined), zero_division=0),
            "F1": f1_score(y_test, search.predict(X_test_combined), average='macro'),
            "Balanced Accuracy": balanced_accuracy_score(y_test, search.predict(X_test_combined))
        })

    # Gesamtes Training auf allen Daten
    X_embeddings_processed = embedding_pipeline.fit_transform(summaries)
    X_combined = np.hstack([X_tabular, X_embeddings_processed])

    search.fit(X_combined, y)
    y_train_pred = search.predict(X_combined),
    y_train_pred_proba = search.predict_proba(X_combined)[:, 1]

    # Calculate training metrics
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

    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, train_metrics, metrics_per_fold


def concat_tab_rte_hgbc(dataset_name, X_tabular, y, nominal_features, summaries, n_splits=3):
    dataset = dataset_name
    ml_method = "HistGradientBoosting"
    emb_method = "Random Trees Embedding"
    concatenation = "yes"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

    embedding_pipeline = Pipeline([
        ("embedding", RandomTreesEmbedding()),
        ("scaler", MinMaxScaler())
    ])
    param_grid = {
            "hist_gb__min_samples_leaf": [5, 10, 15, 20],
            "embedding__n_estimators": [10, 100, 1000],
            "embedding__max_depth": [2, 5, 10, 15],
    }
    search = GridSearchCV(
        estimator=HistGradientBoostingClassifier(categorical_features=nominal_features),
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3)
    )

    for train_index, test_index in skf.split(X_tabular, y):
        # Train-Test-Split für tabellarische und Embedding-Daten
        X_tab_train, X_tab_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        embeddings_train = [summaries[i] for i in train_index]
        embeddings_test = [summaries[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_embeddings_train = embedding_pipeline.fit_transform(embeddings_train)
        X_embeddings_test = embedding_pipeline.transform(embeddings_test)

        # Kombination von tabellarischen Daten und Embeddings
        X_train_combined = np.hstack([X_tab_train, X_embeddings_train])  # todo: Dimension ausgeben und checken
        X_test_combined = np.hstack([X_tab_test, X_embeddings_test])


        search.fit(X_train_combined, y_train)
        y_pred = search.predict(X_test_combined)
        y_pred_proba = search.predict_proba(X_test_combined)[:, 1]

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, search.predict(X_test_combined)).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_per_fold.append({
            "Fold": len(metrics_per_fold),
            "AUC": roc_auc_score(y_test, y_pred_proba),
            "AP": average_precision_score(y_test, y_pred_proba),
            "Sensitivity": recall_score(y_test, y_pred, pos_label=1),
            "Specificity": specificity,
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, average='macro'),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred)
        })

    # Gesamtes Training auf allen Daten
    X_embeddings_processed = embedding_pipeline.fit_transform(summaries)
    X_combined = np.hstack([X_tabular, X_embeddings_processed])
    search.fit(X_combined, y)
    y_pred = search.predict(X_combined)
    y_train_pred_proba = search.predict_proba(X_combined)[:, 1]

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

    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Combined train score: {train_metrics}")
    print(f"Combined test scores: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, train_metrics, metrics_per_fold


def lr_txt_emb_all_emb_agg(feature_extractor, summaries, y, n_splits=3):
    results = {}
    skf = StratifiedKFold(n_splits=n_splits)

    methods = [
        "embedding_cls",
        "embedding_mean_with_cls_and_sep",
        "embedding_mean_without_cls_and_sep"
    ]

    for method in methods:
        test_scores = []
        search = GridSearchCV(
            estimator=Pipeline([
                ("aggregator", EmbeddingAggregator(feature_extractor, method=method)),
                ("numerical_scaler", MinMaxScaler()),
                ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
            ]),
            param_grid={
                "classifier__C": [2, 10, 50, 250]
            },
            scoring="neg_log_loss",
            cv=RepeatedStratifiedKFold(n_splits=3)
        )

        for train_index, test_index in skf.split(summaries, y):
            X_train, X_test = [summaries[i] for i in train_index], [summaries[i] for i in test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit and evaluate for the current method
            search.fit(X_train, y_train)
            y_pred_proba = search.predict_proba(X_test)[:, 1]
            test_scores.append(roc_auc_score(y_test, y_pred_proba))

        # Fit on the entire dataset for train score
        search.fit(summaries, y)
        train_score = roc_auc_score(y, search.predict_proba(summaries)[:, 1])

        # Store results
        results[method] = {
            "train_score": train_score,
            "test_scores": test_scores,
            "best_params": search.best_params_,
            "embedding_size": len(search.best_estimator_.named_steps["classifier"].coef_[0])
        }

        print(f"Method: {method}")
        print(f"Embedding size: {results[method]['embedding_size']}")
        print(f"Best hyperparameters: {results[method]['best_params']}")
        print(f"Train score: {results[method]['train_score']}")
        print(f"Test scores: {results[method]['test_scores']}")
        print("")

    return results

def hgbc_txt_emb_all_emb_agg(feature_extractor, summaries, y, n_splits=3):
    results = {}
    skf = StratifiedKFold(n_splits=n_splits)

    methods = [
        "embedding_cls",
        "embedding_mean_with_cls_and_sep",
        "embedding_mean_without_cls_and_sep"
    ]

    for method in methods:
        test_scores = []

        # Create the GridSearchCV pipeline for this method
        search = GridSearchCV(
            estimator=Pipeline([
                ("aggregator", EmbeddingAggregator(feature_extractor, method=method)),
                ("numerical_scaler", MinMaxScaler()),
                ("hist_gb", HistGradientBoostingClassifier()),
            ]),
            param_grid={
                "hist_gb__min_samples_leaf": [5, 10, 15, 20],
            },
            scoring="neg_log_loss",
            cv=RepeatedStratifiedKFold(n_splits=3)
        )

        # Cross-validation for the current method
        for train_index, test_index in skf.split(summaries, y):
            X_train, X_test = [summaries[i] for i in train_index], [summaries[i] for i in test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train, X_test = np.array(X_train), np.array(X_test)

            # Fit and evaluate
            search.fit(X_train, y_train)
            y_pred_proba = search.predict_proba(X_test)[:, 1]
            test_scores.append(roc_auc_score(y_test, y_pred_proba))

        # Fit on the entire dataset for the train score
        search.fit(np.array(summaries), y)
        train_score = roc_auc_score(y, search.predict_proba(np.array(summaries))[:, 1])

        # Store results
        results[method] = {
           "train_score": train_score,
           "test_scores": test_scores,
           "best_params": search.best_params_,
        }

        # Print results for the current method
        print(f"Method: {method}")
        print(f"Best hyperparameters: {results[method]['best_params']}")
        print(f"Train score: {results[method]['train_score']}")
        print(f"Test scores: {results[method]['test_scores']}")
        print("")

    return results

    # Todo: Comb. mit RTE  | + but test
    # Todo! Wird der beste Aggregierung ausgegeben? | + run both & evaluate

    # Todo: Ergebnisse als Dataframe speichern (to_csv) in csv speichern | weiter

    # Todo: Ausser AUC auch andere Metriken berechnen (Slack) und als Spalten hinzufügen +
    # Todo: check: all previous data displayed correctly?

    # Todo! Dimension Reduktion: Anzahl den Parameter reduzieren; die Daten skalieren (standardscaler),
    #       PCA, Ziel: Overfitting reduzieren ~
    # Todo: Zweite Dimension - Anzahl der Features ausgeben lassen (muss 40 sein) ?

    # Todo: Datei n.1: Test Results no Embedding
    # Spalten für csv-Datei: Dataset, ml_method, emb_method(none), fold, metrics
    # Todo: Datei n.2: Train Results no Embedding
    # Spalten für csv-Datei: Dataset, ml_method, emb_method(none), metrics
    # Todo: Datei n.3: Test Results RT Embedding
    # Spalten für csv-Datei: Dataset, ml_method, emb_method(rt), Concatenation(y/n), fold, metrics
    # Todo: Datei n.4: Train Results RT Embedding
    # Spalten für csv-Datei: Dataset, ml_method, emb_method, Concatenation(y/n), metrics

    # Todo: Datei n.5: Test Results Text Embedding
    # Spalten für csv-Datei: Dataset, ml_method, emb_method(model), Concatenation(y/n), fold, metrics
    # Todo: Datei n.6: Train Results Text Embedding
    # Spalten für csv-Datei: Dataset, ml_method, emb_method(model), Concatenation(y/n), metrics

    # Todo: Pycharm Pro?
