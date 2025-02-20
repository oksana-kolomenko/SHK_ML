import os
import time
import xarray
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding, HistGradientBoostingClassifier
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


def test_text_embeddings(feature_extractor, raw_text_summaries, y, n_splits=3, n_components=None, n_repeats=1):
    embedding_aggregator = EmbeddingAggregator(feature_extractor, method="embedding_cls")
    embeddings = embedding_aggregator.transform(raw_text_summaries)
    print("Generated embeddings shape:", embeddings.shape)
    print(f"Last embedding: {embeddings[-1]}")


def concat_lr_txt_emb_no_pipeline(feature_extractor, raw_text_summaries, X, y, nominal_features, imp_max_iter,
                                  lr_max_iter, text_features, n_repeats, n_splits=3, n_components=40):
    text_emb_method = "embedding_cls"

    # add text as new column
    text_column_name = "text"
    X[text_column_name] = raw_text_summaries

    nominal_imputer = SimpleImputer(strategy="most_frequent")
    nominal_encoder = OneHotEncoder(handle_unknown="ignore")

    numerical_imputer = IterativeImputer(max_iter=imp_max_iter)
    numerical_scaler = MinMaxScaler()

    embedding_aggregator = EmbeddingAggregator(feature_extractor, method=text_emb_method)

    lr_model = LogisticRegression(max_iter=lr_max_iter, class_weight="balanced")

    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    skf.get_n_splits(X, y)

    # prepare data
    numerical_features = list(set(X.columns) - set(nominal_features) - set(text_features))  # todo:replace hardcoding
    print(f"Numerical features identified: {numerical_features}")
    imputed_numerical = numerical_imputer.fit_transform(X[numerical_features])
    scaled_numerical = numerical_scaler.fit_transform(imputed_numerical)

    X[nominal_features] = nominal_imputer.fit_transform(X[nominal_features])
    encoded_nominal = nominal_encoder.fit_transform(X[nominal_features]).toarray()

    # Embeddings
    print(f"Type raw_sum: {type(raw_text_summaries)}")
    print(f"Len raw_sum: {len(raw_text_summaries)}")
    print(f"Type features in column: {type(text_features)}")
    print(f"Len features in column: {len(X[text_features])}")
    print(f"Text features in column: {X[text_features]}")

    raw_embeddings = embedding_aggregator.transform(raw_text_summaries)
    print(f"len raw_embeddings: {len(raw_embeddings)}")
    # print(f"shape raw_embeddings: {raw_embeddings.shape}")

    raw_scalierte_embeddings = numerical_scaler.fit_transform(raw_embeddings)

    print(f"len raw_scal_embeddings: {len(raw_scalierte_embeddings)}")
    print(f"shape raw_scal_embeddings: {raw_scalierte_embeddings.shape}")

    # todo: pre-step: convert X[text] features to a list with the length 82
    embeddings = embedding_aggregator.transform(X[text_column_name].tolist())
    print(f"len embeddings: {len(embeddings)}")  # should be 82
    print(f"shape embeddings: {embeddings.shape}")

    scalierte_embeddings = numerical_scaler.fit_transform(embeddings)

    print(f"len scal_embeddings: {len(scalierte_embeddings)}")
    print(f"shape scal_embeddings: {scalierte_embeddings.shape}")  # should be(82, 768)
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(scalierte_embeddings)

    combined_features = np.hstack((encoded_nominal, scaled_numerical, reduced_embeddings))

    print("Final Scaled Data (as NumPy Array):")
    print(combined_features)

    print(f"Categorical Features: {len(nominal_features)}")
    print(f"Text Features: {len(text_features)}")
    print(f"Numerical Features: {len(set(X.columns) - set(nominal_features))}")

    # Count unique categories in each categorical feature
    for col in nominal_features:
        print(f"{col} has {X[col].nunique()} unique categories.")

    # Embeddings
    # pca = PCA(n_components=n_components)
    # reduced_embeddings = pca.fit_transform(scalierte_embeddings)

    # explained_variance = pca.explained_variance_ratio_
    # print(f"Explained Variance by Top {n_components} Components: {sum(explained_variance) * 100:.2f}%")

    print(f"Type of embeddings: {type(embeddings)}")
    print(f"Combined_X shape: {combined_features.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    # print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

    print(f"Combined shape: {combined_features.shape}")

    test_scores = []

    for train_index, test_index in skf.split(combined_features, y):
        X_train, X_test = combined_features[train_index], combined_features[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print("\nTrain Indices:", train_index)
        print("Test Indices:", test_index)
        print("X_train:\n", X_train)
        print("X_test:\n", X_test)
        print("y_train:", y_train)
        print("y_test:", y_test)
        lr_model.fit(X_train, y_train)

        score = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
        test_scores.append(score)
        print(score)

    lr_model.fit(combined_features, y)

    score_ = roc_auc_score(y, lr_model.predict_proba(combined_features)[:, 1])

    print(f"LR_model train Scores: {score_}")
    print(f"LR_model test Scores: {test_scores}")


def lr_txt_emb_pca_no_pipeline(feature_extractor, raw_text_summaries, y, nominal_features, n_splits=3,
                               n_components=40, n_repeats=1):
    text_emb_method = "embedding_cls"
    nominal_imputer = SimpleImputer(strategy="most_frequent")
    numerical_imputer = IterativeImputer(max_iter=5)
    numerical_scaler = MinMaxScaler()
    embedding_aggregator = EmbeddingAggregator(feature_extractor, method=text_emb_method)
    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    # Embeddings
    embeddings = embedding_aggregator.transform(raw_text_summaries)
    scalierte_embeddings = numerical_scaler.fit_transform(embeddings)
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(scalierte_embeddings)

    explained_variance = pca.explained_variance_ratio_
    print(f"Explained Variance by Top {n_components} Components: {sum(explained_variance) * 100:.2f}%")
    print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

    test_scores = []

    for train_index, test_index in skf.split(reduced_embeddings, y):
        X_train, X_test = reduced_embeddings[train_index], reduced_embeddings[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print("\nTrain Indices:", train_index)
        print("Test Indices:", test_index)
        print("X_train:\n", X_train)
        print("X_test:\n", X_test)
        print("y_train:", y_train)
        print("y_test:", y_test)
        lr_model.fit(X_train, y_train)

        score = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
        test_scores.append(score)
        print(score)

    lr_model.fit(reduced_embeddings, y)

    score_ = roc_auc_score(y, lr_model.predict_proba(reduced_embeddings)[:, 1])

    print(f"LR_model train Scores: {score_}")
    print(f"LR_model test Scores: {test_scores}")


def logistic_regression(dataset_name, X, y, nominal_features, n_repeats=10, n_splits=3, n_components=None):
    # Todo: try encoding categ. features with OHE (after finding all categ. features (with Ricardo))
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
                    ("numerical_imputer", IterativeImputer(max_iter=50)),  # todo=5 fürs Testen
                    ("numerical_scaler", MinMaxScaler())
                ] + ([pca_step] if pca_step else [])), numerical_features),
            ])),
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
        ]),
        param_grid={"classifier__C": [2, 10, 50, 250]},
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
            "embedding__n_estimators": [10, 100, 1000],
            "embedding__max_depth": [2, 5, 10, 15],
            "classifier__C": [2, 10, 50, 250]
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
    pca_components = f"PCA ({n_components} components)" if n_components else "none"

    pipeline_steps = [
        ("aggregator", EmbeddingAggregator(feature_extractor))
    ]
    if n_components:
        pipeline_steps.append(("numerical_scaler", StandardScaler()))
        pipeline_steps.append(("pca", PCA(n_components=n_components)))
        #pipeline_steps.append(("numerical_scaler", MinMaxScaler())) #todo: needed here?
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
        print(f"Number of samples (train) (n_samples): {n_samples}") #
        print(f"Number of samples (test) (n_samples): {len(X_test)}") #
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

    best_params = f"{search.best_params_}" # is only for training data!

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

    # Define GridSearchCV once before the loop
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
    print(f"HGBC train fitting... ")
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

    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, best_params, train_metrics, metrics_per_fold

"""
def hgbc_ran_tree_emb(dataset_name, X, y, nominal_features, n_splits=3):
    dataset = dataset_name
    ml_method = "HistGradientBoosting"
    emb_method = "random tree embedding"
    concatenation = "no"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)
    categorical_feature_indices = [X.columns.get_loc(col) for col in nominal_features]
    dense_transformer = FunctionTransformer(lambda x: x.toarray() if hasattr(x, "toarray") else x)

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]), nominal_features),
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=5)) # todo only test!
                    # TODO: skalieren (nein)
                ]), list(set(X.columns.values) - set(nominal_features))),
            ] )),
            #("to_dense", dense_transformer),
            ("embedding", RandomTreesEmbedding(random_state=42)),
            ("hist_gb", HistGradientBoostingClassifier(categorical_features=categorical_feature_indices))
        ]),
        param_grid={
            "embedding__n_estimators": [10], #, 100, 1000],todo only test!
            "embedding__max_depth": [2, 5], #, 10, 15],todo only test!
            "hist_gb__min_samples_leaf": [5] #, 10, 15, 20]todo only test!
        },
        scoring="neg_log_loss",
        # cv=RepeatedStratifiedKFold(n_splits=3, random_state=42)
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42) #todo only test!
    )

    # Metrics for each fold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Get the transformer pipeline from GridSearchCV before fitting
        transformer = search.estimator.named_steps['transformer']

        # Transform X_train and X_test BEFORE calling `search.fit()`
        X_train_transformed = transformer.fit_transform(X_train)
        X_test_transformed = transformer.transform(X_test)

        # Convert sparse matrices to dense if necessary
        if hasattr(X_train_transformed, "toarray"):
            X_train_transformed = X_train_transformed.toarray()
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()

        # Now fit GridSearchCV with transformed data
        search.fit(X_train_transformed, y_train)

        # Fit the model for each fold
        # search.fit(X_train, y_train)

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
    best_params = f"{search.best_params_}"
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, best_params, train_metrics, metrics_per_fold
"""



def hgbc_rte(X, y, nominal_features, n_splits=3):  # todo
    hgbc_rt_emb_test_scores = []
    skf = StratifiedKFold(n_splits=n_splits)
    categorical_feature_indices = [X.columns.get_loc(col) for col in nominal_features]
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        search = GridSearchCV(
            estimator=Pipeline([
                ("transformer", ColumnTransformer([
                    ("nominal", Pipeline([
                        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                        ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                    ]), nominal_features),
                    ("numerical", Pipeline([
                        ("numerical_imputer", IterativeImputer(max_iter=5))#30))
                    ]), list(set(X_train.columns.values) - set(nominal_features))),
                ])),
                ("embedding", RandomTreesEmbedding()),
                ("hist_gb", HistGradientBoostingClassifier())#categorical_features=categorical_feature_indices))
            ]),
            param_grid={
                # extended hyperparameter search for better embedding
                "embedding__n_estimators": [10],  # , 100, 1000],
                "embedding__max_depth": [2, 5],  # , 10, 15],
                "hist_gb__min_samples_leaf": [5]  # , 10, 15, 20]
            },
            scoring="neg_log_loss",
            cv=3
        )
        print(f"X_train size before: {X_train.shape}")
        X_train_xr = X_train.to_xarray()
        print(f"X_train size after: {X_train_xr}")
        print(f"y_train size: {len(y_train)}")
        search.fit(X_train, y_train)
        hgbc_rt_emb_test_scores.append(roc_auc_score(y_test, search.predict_proba(X_test)[:, 1]))

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), nominal_features),
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=5)) #30))
                ]), list(set(X_train.columns.values) - set(nominal_features))),
            ])),
            ("embedding", RandomTreesEmbedding()),
            ("hist_gb", HistGradientBoostingClassifier(categorical_features=nominal_features)) # todo: brauche ich es hier?
        ]),
        param_grid={
            # extended hyperparameter search for better embedding
            "embedding__n_estimators": [10], #, 100, 1000],
            "embedding__max_depth": [2, 5], #, 10, 15],
            "hist_gb__min_samples_leaf": [5] #, 10, 15, 20]
        },
        scoring="neg_log_loss",
        cv=3
    )
    search.fit(X.to_xarray(), y)
    hgbc_rt_emb_train_score = roc_auc_score(y, search.predict_proba(X)[:, 1])
    print(f"embedding size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"best hyperparameters: {search.best_params_}")
    print(f"lr_ran_tree_emb_train_score: {hgbc_rt_emb_train_score}")
    print(f"lr_ran_tree_emb_test_scores: {hgbc_rt_emb_test_scores}")

    return hgbc_rt_emb_train_score, hgbc_rt_emb_test_scores


def hgbc_txt_emb(dataset_name, emb_method, feature_extractor, summaries, y,
                 n_components, n_repeats, n_splits=3):
    dataset = dataset_name
    ml_method = "HistGradientBoosting"
    emb_method = emb_method
    concatenation = "no"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)

    pca_components = f"PCA ({n_components} components)" if n_components else "none"

    pipeline_steps = [
        ("aggregator", EmbeddingAggregator(feature_extractor))
    ]
    if n_components:
        pipeline_steps.append(("numerical_scaler", StandardScaler()))
        pipeline_steps.append(("pca", PCA(n_components=n_components))) # todo: danach skalieren?

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

    print(f"Numerical features identified: {numerical_features}")
    print(f"Setting up the pipeline at {time.strftime('%H:%M:%S', time.localtime(time.time()))}")
    print(f"Tabelle Größe {X_tabular.shape}")  # muss 41X82
    print(f"All columns: {X_tabular.columns}")

    pca_components = f"PCA ({n_components} components)" \
        if n_components else "none"

    pipeline_text_steps = [
        ("embedding_aggregator", EmbeddingAggregator(feature_extractor)),
    ]
    if n_components:
        pipeline_text_steps.append(("numerical_scaler", StandardScaler()))
        pipeline_text_steps.append(("pca", PCA(n_components=n_components)))
    else:
        pipeline_text_steps.append(("numerical_scaler", MinMaxScaler()))

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), nominal_features),
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
                    ("numerical_scaler", MinMaxScaler())
                ]), numerical_features),
                ("text", Pipeline(pipeline_text_steps), text_features),
            ])),
            # todo: ohne classificator die shape nach der vorverarbeitung ausgeben lassen
            # todo: muss ca [40] + [768] sein
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
        cv=RepeatedStratifiedKFold(n_splits=3)
    )

    for train_index, test_index in skf.split(X_tabular, y):
        # Split tabular data, summaries, and labels
        X_train, X_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
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


# Problem: after RTE there are 40 Features, but when debugging ~ 2000
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

    numerical_features = list(set(X_tabular.columns) - set(nominal_features))

    print(f"Nominal Features: {nominal_features}")
    print(f"Nominal Features len: {len(nominal_features)}")
    print(f"Numerical Features: {numerical_features}")
    print(f"Numerical Features len: {len (numerical_features)}")
    print(f"X_tabular Columns: {list(X_tabular.columns)}")
    print(f"X_tabular Columns len: {len(list(X_tabular.columns))}")

    pipeline = Pipeline([
        ("feature_combiner", FeatureUnion([
            # Verarbeitung der tabellarischen Daten
            ("raw", ColumnTransformer([
                ("nominal", Pipeline([
                    ("debug_nominal", DebugTransformer(name="Nominal Debug")), # 5
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore")), #14
                    ("debug_nominal_after", DebugTransformer(name="Nominal Debug after"))
                ]), nominal_features),
                ("numerical", Pipeline([
                    ("debug_numerical", DebugTransformer(name="Numerical Debug")),
                    ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)), # 35
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
                    ("numerical", Pipeline([
                        ("debug_numerical", DebugTransformer(name="Numerical Debug Emb")),
                        ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
                        ("debug_numerical_after", DebugTransformer(name="Numerical Debug Emb after"))
                    ]), numerical_features),
                ], remainder="passthrough")),
                ("debug_embedding", DebugTransformer(name="Embedding Debug")),
                ("embedding", RandomTreesEmbedding(random_state=42)),  # check
                ("debug_embedding_after", DebugTransformer(name="Embedding Debug after"))
            ]))
        ])),
        ("debug_final", DebugTransformer(name="Final Feature Set")),
        ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=class_max_iter))
    ])

    param_grid = {
        "classifier__C": [2, 10, 50, 250],
        "feature_combiner__embeddings__embedding__n_estimators": [10, 100, 1000], # todo: mit anderen Parameter (z.B. 100) testen!
        "feature_combiner__embeddings__embedding__max_depth": [2, 5, 10, 15],
    }
    # Ensure the feature_combiner is fitted before transformation
    #pipeline.named_steps["feature_combiner"].fit(X_tabular, y) # ~2000

    # Manually transform data for debugging
    #X_transformed = pipeline.named_steps["feature_combiner"].transform(X_tabular)

    # Print out shape of final transformed features
   # print(f"Final Feature Set Shape (Before Model Training): {X_transformed.shape}") # ~2000

    # Check if any feature vector is empty
   # if X_transformed.shape[1] == 0:
   #     raise ValueError("🚨 FeatureUnion is not concatenating features correctly! Feature set is empty.")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=n_repeats)
    )

    # Cross-Validation
    for train_index, test_index in skf.split(X_tabular, y):
        # Aufteilen der tabellarischen Daten und Embeddings
        X_tab_train, X_tab_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model
        search.fit(X_tab_train, y_train)

        # Predictions and probabilities
        y_test_pred = search.predict(X_tab_test)
        y_test_pred_proba = search.predict_proba(X_tab_test)[:, 1]

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
    search.fit(X_tabular, y)
    y_train_pred = search.predict(X_tabular)
    y_train_pred_proba = search.predict_proba(X_tabular)[:, 1]

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

    best_params = f"{search.best_params_}"

    print(f"Feature set size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, best_params, pca_n_comp, train_metrics, metrics_per_fold


# läuft
def concat_txt_tab_hgbc(dataset_name, emb_method,
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

    # separate numerical features
    non_text_columns = list(set(X_tabular.columns) -
                            set(text_features))

    print(f"All columns length: {X_tabular.shape}")
    print(f"Non-text columns length: {len(X_tabular[non_text_columns])}")

    categorical_indices = [X_tabular.columns.get_loc(col) for col in nominal_features]

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("text", Pipeline([
                    ("embedding_aggregator", EmbeddingAggregator(feature_extractor)),
                    ("debug_text", DebugTransformer(name="Text Debug")),
                    ("numerical_scaler", MinMaxScaler())
                ]), text_features)
            ])),
            ("classifier", HistGradientBoostingClassifier(categorical_features=categorical_indices))
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
        cv=RepeatedStratifiedKFold(n_splits=3)
    )

    for train_index, test_index in skf.split(X_tabular, y):
        # Todo: Problem, if text feature is not in the split then the split have very different sizes
        X_train, X_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f"Length of X_tab_train: {len(X_train)}")
        print(f"Length of y_train: {len(y_train)}")

        assert len(X_train) == len(y_train), "Mismatch in training data sizes"

        # Modelltraining und Bewertung
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

    print(f"X_tabular len: {len(X_tabular)}")
    print(f"Text_features len: {len(text_features)}")  # ist = 1 muss aber 82 sein
    print(f"y len: {len(y)}")
    assert len(X_tabular) == len(y), "Mismatch in training data sizes"

    search.fit(X_tabular, y)
    y_train_pred = search.predict(X_tabular)
    y_train_pred_proba = search.predict_proba(X_tabular)[:, 1]

    print(f"X_tabular shape {X_tabular.shape}")
    print(f"y shape: {y.shape}")  # Should be (82,)
    print(f"y_train_pred shape: {y_train_pred.shape}")  # Should also be (82,)

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

    best_params = f"{search.best_params_}"
    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return (dataset, ml_method, emb_method, concatenation, best_params,
            n_components, train_metrics, metrics_per_fold)


def concat_hgbc_rte(dataset_name, X_tabular, y, nominal_features,
                    pca_n_comp, n_splits=3):
    dataset = dataset_name
    ml_method = "HistGradientBoosting"
    emb_method = "Random Trees Embedding"
    concatenation = "yes"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

    # preprocess table data as usual
    pipeline_table = ("raw", ColumnTransformer([
                         ("nominal", Pipeline([
                             ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                             ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                         ]), nominal_features),
                         ("numerical", Pipeline([
                             ("numerical_imputer", IterativeImputer(max_iter=5)) # todo: 30))
                         ]), list(set(X_tabular.columns.values) - set(nominal_features))),
                     ], remainder="passthrough"))

    pipeline_rte = ("embeddings", Pipeline([
                ("transformer", ColumnTransformer([
                    ("nominal", Pipeline([
                        ("debug_nominal_emb", DebugTransformer(name="Nominal Debug Emb after")),
                        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                        ("nominal_encoder", OneHotEncoder(handle_unknown="ignore")),
                        ("debug_nominal_emb_after", DebugTransformer(name="Nominal Debug Emb"))
                    ]), nominal_features),
                    ("numerical", Pipeline([
                        ("debug_numerical", DebugTransformer(name="Numerical Debug Emb")),
                        ("numerical_imputer", IterativeImputer(max_iter=5)), # todo: 30))
                        ("debug_numerical_after", DebugTransformer(name="Numerical Debug Emb after"))
                    ]), list(set(X_tabular.columns.values) - set(nominal_features))),
                ], remainder="passthrough")),
                ("debug_embedding", DebugTransformer(name="Embedding Debug")),
                ("embedding", RandomTreesEmbedding(random_state=42)),
                ("debug_embedding_after", DebugTransformer(name="Embedding Debug after"))
            ]))


    param_grid = {
        "hist_gb__min_samples_leaf": [5, 10, 15, 20],
        "embedding__n_estimators": [10, 100, 1000],
        "embedding__max_depth": [2, 5, 10, 15],
    }
    search = GridSearchCV(
        estimator=Pipeline([
            ("feature_combiner", FeatureUnion([
                pipeline_table,
                pipeline_rte
            ])),
            ("embedding", RandomTreesEmbedding()),
            ("hist_gb", HistGradientBoostingClassifier(categorical_features=nominal_features)) # todo: brauche ich es hier?
        ]),
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3)
    )

    for train_index, test_index in skf.split(X_tabular, y):
        # Train-Test-Split für tabellarische und Embedding-Daten
        X_train, X_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # todo big TODO
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

    # Gesamtes Training auf allen Daten
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


class DebugTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, name="Step"):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"{self.name}: Input shape {X.shape}")  # Print before transformation
        if isinstance(X, pd.DataFrame):
            X_transformed = X.to_numpy()  # Convert to NumPy array
        else:
            X_transformed = X
        print(f"{self.name}: Output shape {X_transformed.shape}")  # Print after transformation
        return X_transformed


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
