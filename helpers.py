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
    FunctionTransformer, LabelEncoder  # , OrdinalEncoder
# from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, recall_score, precision_score, f1_score,
                             balanced_accuracy_score, confusion_matrix, average_precision_score)
from text_emb_aggregator import EmbeddingAggregator
from values import DATASET_CONFIGS, DatasetName


def calc_metrics(y, y_pred, y_pred_proba):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        "AUC": roc_auc_score(y, y_pred_proba),
        "AP": average_precision_score(y, y_pred_proba),
        "Sensitivity": recall_score(y, y_pred, pos_label=1),
        "Specificity": specificity,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, average='macro'),
        "Balanced Accuracy": balanced_accuracy_score(y, y_pred)
    }

    return metrics


def train(search, X_train, y_train, X_test, y_test):
    search.fit(X_train, y_train)

    y_test_pred = search.predict(X_test)
    y_test_pred_proba = search.predict_proba(X_test)[:, 1]

    metrics = calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba)

    return metrics


def logistic_regression(dataset_name, X, y, nominal_features, pca):
    y = pd.Series(y)

    print(f"[INFO] Rows before NaN removal: X={len(X)}, y={len(y)}")
    print(f"[INFO] NaNs in y before removal: {y.isna().sum()}")

    valid_indices = ~y.isna()
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    print(f"[INFO] Rows after NaN removal: X={len(X)}, y={len(y)}")
    print(f"[INFO] NaNs in y after removal: {y.isna().sum()}")

    if not np.issubdtype(y.dtype, np.number):
        print(f"Label encoding: {y.unique()}")
        le = LabelEncoder()
        y = le.fit_transform(y)  # this returns a NumPy array
    else:
        y = y.to_numpy()

    config = DATASET_CONFIGS[dataset_name]
    n_splits, n_repeats, n_components = config.splits, config.n_repeats, config.pca if pca else None

    ml_method, emb_method, concatenation, pca_components =\
        "logistic regression", "none", "no", f"PCA ({n_components} components)" if pca else "none"

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
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
                ]), nominal_features),
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=50)),
                    ("numerical_scaler", MinMaxScaler())
                ] + ([pca_step] if pca_step else [])), numerical_features),
            ])),
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
        ]),
        param_grid={"classifier__C": [2, 10]},
        # param_grid={"classifier__C": [0.1, 2, 10, 100]},
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    )

    if dataset_name == DatasetName.POSTTRAUMA.value:
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print(f"Log reg test fitting... ")
            search.fit(X_train, y_train)

            y_test_pred = search.predict(X_test)
            y_test_pred_proba = search.predict_proba(X_test)[:, 1]

            metrics_per_fold.append(
                calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    elif dataset_name == DatasetName.CYBERSECURITY.value or dataset_name == DatasetName.LUNG_DISEASE.value:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        metrics_per_fold.append(
            calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    else:
        print(f"Unknown dataset {dataset_name}")

    print(f"Log reg train fitting... ")
    search.fit(X, y)
    y_train_pred = search.predict(X)
    y_train_pred_proba = search.predict_proba(X)[:, 1]

    # Training metrics
    train_metrics = calc_metrics(y=y, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba)

    best_params = f"{search.best_params_}"

    print(f"Feature set size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset_name, ml_method, emb_method, concatenation, best_params, pca_components, train_metrics, metrics_per_fold


def lr_rte(dataset_name, X, y, nominal_features, pca):
    y = pd.Series(y)

    print(f"[INFO] Rows before NaN removal: X={len(X)}, y={len(y)}")
    print(f"[INFO] NaNs in y before removal: {y.isna().sum()}")

    valid_indices = ~y.isna()
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    print(f"[INFO] Rows after NaN removal: X={len(X)}, y={len(y)}")
    print(f"[INFO] NaNs in y after removal: {y.isna().sum()}")

    if not np.issubdtype(y.dtype, np.number):
        print(f"Label encoding: {y.unique()}")
        le = LabelEncoder()
        y = le.fit_transform(y)  # this returns a NumPy array
    else:
        y = y.to_numpy()

    dataset = dataset_name
    config = DATASET_CONFIGS[dataset]
    n_splits = config.splits
    #n_components = config.pca if pca else None
    n_repeats = config.n_repeats

    ml_method = "logistic regression"
    emb_method = "RTE"
    concatenation = "no"
    #pca_components = f"PCA ({n_components} components)" if n_components else "none"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                # Encode nominal features with OHE
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
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
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    )
    if dataset_name == DatasetName.POSTTRAUMA.value:
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print(f"Log Reg rte test fitting... ")

            # Fit the model for each fold
            search.fit(X_train, y_train)

            y_test_pred = search.predict(X_test)
            y_test_pred_proba = search.predict_proba(X_test)[:, 1]

            metrics_per_fold.append(
                calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    elif dataset_name == DatasetName.CYBERSECURITY.value or dataset_name == DatasetName.LUNG_DISEASE.value:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        metrics_per_fold.append(
            calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    else:
        print(f"Unknown dataset {dataset_name}")

    # Train the final model on the full dataset
    print(f"Log reg rte train fitting... ")
    search.fit(X, y)
    y_train_pred = search.predict(X)
    y_train_pred_proba = search.predict_proba(X)[:, 1]

    # Training metrics
    train_metrics = calc_metrics(y=y, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba)

    best_params = f"{search.best_params_}"
    print(f"Embedding size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, best_params, train_metrics, metrics_per_fold


# n_components aus dem Datensatz nehmen (40 für Posttrauma (shape[1])
def lr_txt_emb(dataset_name, emb_method, feature_extractor, raw_text_summaries, y, max_iter, pca):
    y = pd.Series(y)

    print(f"[INFO] Rows before NaN removal: X={len(raw_text_summaries)}, y={len(y)}")
    print(f"[INFO] NaNs in y before removal: {y.isna().sum()}")

    raw_text_summaries = pd.Series(raw_text_summaries)

    valid_indices = ~y.isna()
    raw_text_summaries = raw_text_summaries.loc[valid_indices]
    y = y.loc[valid_indices]

    print(f"[INFO] Rows after NaN removal: X={len(raw_text_summaries)}, y={len(y)}")
    print(f"[INFO] NaNs in y after removal: {y.isna().sum()}")

    if not np.issubdtype(y.dtype, np.number):
        print(f"Label encoding: {y.unique()}")
        le = LabelEncoder()
        y = le.fit_transform(y)  # this returns a NumPy array
    else:
        y = y.to_numpy()

    config = DATASET_CONFIGS[dataset_name]
    n_splits = config.splits
    n_components = config.pca
    n_repeats = config.n_repeats

    ml_method = "logistic regression"
    concatenation = "no"
    metrics_per_fold = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    is_sentence_transformer = any(
        key in emb_method.lower()
        for key in ["gtr-t5-base", "sentence-t5-base", "modernbert_embed"]
    )

    pca_components = f"PCA ({n_components} components)" if pca else "none"

    pipeline_steps = [
        ("aggregator", EmbeddingAggregator(
            feature_extractor=feature_extractor,
            is_sentence_transformer=is_sentence_transformer
        ))
    ]

    if pca:
        pipeline_steps.append(("numerical_scaler", StandardScaler()))
        pipeline_steps.append(("pca", PCA(n_components=n_components)))
        # pipeline_steps.append(("numerical_scaler", MinMaxScaler()))
    else:
        pipeline_steps.append(("numerical_scaler", MinMaxScaler()))

    pipeline_steps.append(("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=max_iter)))

    search = GridSearchCV(
        estimator=Pipeline(pipeline_steps),
        param_grid={
            "classifier__C": [2, 10],
            "aggregator__method": [
                "embedding_cls",
                "embedding_mean_with_cls_and_sep",
                "embedding_mean_without_cls_and_sep"
            ]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    )

    # === Evaluation ===
    if dataset_name == DatasetName.POSTTRAUMA.value:
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

            best_param = f"Best params for this fold: {search.best_params_}"
            print(best_param)

            metrics_per_fold.append(
                calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    elif dataset_name == DatasetName.CYBERSECURITY.value or dataset_name == DatasetName.LUNG_DISEASE.value:
        X_train, X_test, y_train, y_test = train_test_split(raw_text_summaries, y, test_size=0.2, random_state=42)

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Fit and evaluate
        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        best_param = f"Best params for this fold: {search.best_params_}"
        print(best_param)

        metrics_per_fold.append(
            calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    search.fit(
        raw_text_summaries,
        y
    )

    # change to X_train instead of all data for cybersec & lungdisease
    y_train_pred = search.predict(raw_text_summaries)
    y_train_pred_proba = search.predict_proba(raw_text_summaries)[:, 1]

    # Training metrics
    train_metrics = calc_metrics(y=y, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba)

    best_params = f"{search.best_params_}"

    print(f"embedding size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset_name, ml_method, emb_method, concatenation, best_params, pca_components, train_metrics, metrics_per_fold


def hgbc(dataset_name, X, y, nominal_features, pca):
    y = pd.Series(y)

    print(f"[INFO] Rows before NaN removal: X={len(X)}, y={len(y)}")
    print(f"[INFO] NaNs in y before removal: {y.isna().sum()}")

    valid_indices = ~y.isna()
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    print(f"[INFO] Rows after NaN removal: X={len(X)}, y={len(y)}")
    print(f"[INFO] NaNs in y after removal: {y.isna().sum()}")

    if not np.issubdtype(y.dtype, np.number):
        print(f"Label encoding: {y.unique()}")
        le = LabelEncoder()
        y = le.fit_transform(y)  # this returns a NumPy array
    else:
        y = y.to_numpy()

    config = DATASET_CONFIGS[dataset_name]
    n_splits, n_repeats, n_components = config.splits, config.n_repeats, config.pca if pca else None

    ml_method, emb_method, concatenation, pca_components =\
        "HGBC", "none", "no", f"PCA ({n_components} components)" if pca else "none"

    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)

    pca_step = ("pca", PCA(n_components=n_components)) if n_components else None

    search = GridSearchCV(
        estimator=Pipeline([
            ("hist_gb", HistGradientBoostingClassifier(categorical_features=nominal_features))
        ]),
        param_grid={"hist_gb__min_samples_leaf": [5, 10, 15, 20]},
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    )

    if dataset_name == DatasetName.POSTTRAUMA.value:
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            metrics = train(search=search, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

            metrics_per_fold.append(metrics)

    elif dataset_name == DatasetName.CYBERSECURITY.value or dataset_name == DatasetName.LUNG_DISEASE.value:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        metrics = train(search=search, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        metrics_per_fold.append(metrics)

    else:
        print(f"Unknown dataset {dataset_name}")

    print(f"HGBC train fitting... ")
    search.fit(X, y)
    y_train_pred = search.predict(X)
    y_train_pred_proba = search.predict_proba(X)[:, 1]

    train_metrics = calc_metrics(y=y, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba)
    best_params = f"{search.best_params_}"

    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset_name, ml_method, emb_method, concatenation, best_params, pca_components, train_metrics, metrics_per_fold


def hgbc_rte(dataset_name, X, y, nominal_features):
    y = pd.Series(y)

    print(f"[INFO] Rows before NaN removal: X={len(X)}, y={len(y)}")
    print(f"[INFO] NaNs in y before removal: {y.isna().sum()}")

    valid_indices = ~y.isna()
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    print(f"[INFO] Rows after NaN removal: X={len(X)}, y={len(y)}")
    print(f"[INFO] NaNs in y after removal: {y.isna().sum()}")

    if not np.issubdtype(y.dtype, np.number):
        print(f"Label encoding: {y.unique()}")
        le = LabelEncoder()
        y = le.fit_transform(y)  # this returns a NumPy array
    else:
        y = y.to_numpy()

    dataset = dataset_name
    config = DATASET_CONFIGS[dataset]
    n_splits = config.splits
    n_components = config.pca
    n_repeats = config.n_repeats

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
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
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
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    )

    if dataset_name == DatasetName.POSTTRAUMA.value:
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print(f"X_train size before: {X_train.shape}")
            print(f"y_train size: {len(y_train)}")
            search.fit(X_train, y_train)

            y_test_pred = search.predict(X_test)
            y_test_pred_proba = search.predict_proba(X_test)[:, 1]

            metrics_per_fold.append(
                calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    elif dataset_name == DatasetName.CYBERSECURITY.value or dataset_name == DatasetName.LUNG_DISEASE.value:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        metrics_per_fold.append(
            calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    else:
        print(f"Unknown dataset {dataset_name}")

    search.fit(X, y)
    y_train_pred = search.predict(X)
    y_train_pred_proba = search.predict_proba(X)[:, 1]

    train_metrics = calc_metrics(y=y, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba)

    best_params = f"{search.best_params_}"

    hgbc_rt_emb_train_score = roc_auc_score(y, search.predict_proba(X)[:, 1])
    print(f"best hyperparameters: {best_params}")
    print(f"lr_ran_tree_emb_train_score: {hgbc_rt_emb_train_score}")

    return dataset_name, ml_method, emb_method, conc, best_params, train_metrics, metrics_per_fold


def hgbc_txt_emb(dataset_name, emb_method, feature_extractor, summaries, y, pca):
    print(f"Started: hgbc_txt_emb with {feature_extractor}")
    y = pd.Series(y)

    print(f"[INFO] Rows before NaN removal: X={len(summaries)}, y={len(y)}")
    print(f"[INFO] NaNs in y before removal: {y.isna().sum()}")

    summaries = pd.Series(summaries)

    valid_indices = ~y.isna()
    summaries = summaries.loc[valid_indices]
    y = y.loc[valid_indices]

    print(f"[INFO] Rows after NaN removal: X={len(summaries)}, y={len(y)}")
    print(f"[INFO] NaNs in y after removal: {y.isna().sum()}")

    if not np.issubdtype(y.dtype, np.number):
        print(f"Label encoding: {y.unique()}")
        le = LabelEncoder()
        y = le.fit_transform(y)  # this returns a NumPy array
    else:
        y = y.to_numpy()
    config = DATASET_CONFIGS[dataset_name]
    n_splits = config.splits
    n_components = config.pca
    n_repeats = config.n_repeats

    ml_method = "HistGradientBoosting"
    emb_method = emb_method
    concatenation = "no"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)
    is_sentence_transformer = False
    if "gtr_t5_base" in emb_method.lower() or "sentence_t5_base" in emb_method.lower() or "modernbert_embed" in emb_method.lower():
        is_sentence_transformer = True

    pca_components = f"PCA ({n_components} components)" if pca else "none"

    pipeline_steps = [
        ("aggregator", EmbeddingAggregator(
            feature_extractor=feature_extractor,
            is_sentence_transformer=is_sentence_transformer))
    ]
    if pca:
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
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    )
    print(f"len of summaries: {len(summaries)}")
    print(f"len of y: {len(y)}")

    # === Evaluation ===
    if dataset_name == DatasetName.POSTTRAUMA.value:
        for train_index, test_index in skf.split(summaries, y):
            X_train, X_test = [summaries[i] for i in train_index], [summaries[i] for i in test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train, X_test = np.array(X_train), np.array(X_test)

            metrics = train(search=search, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

            metrics_per_fold.append(metrics)

    elif dataset_name == DatasetName.CYBERSECURITY.value or dataset_name == DatasetName.LUNG_DISEASE.value:
        X_train, X_test, y_train, y_test = train_test_split(summaries, y, test_size=0.2, random_state=42)

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Fit and evaluate
        search.fit(np.array(X_train), y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        best_param = f"Best params for this fold: {search.best_params_}"
        print(best_param)

        metrics_per_fold.append(
            calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    # train on the full dataset
    search.fit(
        np.array(summaries),
        y
    )

    y_train_pred = search.predict(np.array(summaries))
    y_train_pred_proba = search.predict_proba(np.array(summaries))[:, 1]

    # Training metrics
    train_metrics = calc_metrics(y=y, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba)
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

    return dataset_name, ml_method, emb_method, concatenation, best_params, pca_components, train_metrics, metrics_per_fold


# läuft
def concat_lr_txt_emb(dataset_name, emb_method,
                      feature_extractor, raw_text_summaries,
                      X_tabular, y, nominal_features, text_feature_column_name,
                      imp_max_iter, class_max_iter, concatenation, pca):
    start_time = time.time()
    readable_time = time.strftime("%H:%M:%S", time.localtime(start_time))
    print(f"Starting the concat_lr_txt_emb method {readable_time}")
    y = pd.Series(y)

    if not np.issubdtype(y.dtype, np.number):
        print(f"Label encoding: {y.unique()}")
        le = LabelEncoder()
        y = le.fit_transform(y)  # this returns a NumPy array
    else:
        y = y.to_numpy()

    dataset = dataset_name
    config = DATASET_CONFIGS[dataset]
    n_splits = config.splits
    n_components = config.pca
    n_repeats = config.n_repeats

    ml_method = "Logistic Regression"
    emb_method = emb_method
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
        if pca else "none"

    is_sentence_transformer = False
    if ("gtr_t5_base" in emb_method.lower() or "sentence_t5_base" in emb_method.lower()
            or "modernbert_embed" in emb_method.lower()):
        is_sentence_transformer = True

    pipeline_text_steps = [
        ("embedding_aggregator", EmbeddingAggregator(
            feature_extractor=feature_extractor,
            is_sentence_transformer=is_sentence_transformer
        )),
    ]
    if pca:
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
            "classifier__C": [2, 10],
            "transformer__text__embedding_aggregator__method": [
                "embedding_cls",
                "embedding_mean_with_cls_and_sep",
                "embedding_mean_without_cls_and_sep"
            ]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    )
    # === Evaluation ===
    if dataset_name == DatasetName.POSTTRAUMA.value:
        for train_index, test_index in skf.split(X_tabular, y):
            X_train, X_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            search.fit(X_train, y_train)

            y_test_pred = search.predict(X_test)
            y_test_pred_proba = search.predict_proba(X_test)[:, 1]

            metrics_per_fold.append(
                calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    elif dataset_name == DatasetName.CYBERSECURITY.value or dataset_name == DatasetName.LUNG_DISEASE.value:
        X_train, X_test, y_train, y_test = train_test_split(X_tabular, y, test_size=0.2, random_state=42)

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Fit and evaluate
        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        best_param = f"Best params for this fold: {search.best_params_}"
        print(best_param)

        metrics_per_fold.append(
            calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    search.fit(X_tabular, y)

    y_train_pred = search.predict(X_tabular)
    y_train_pred_proba = search.predict_proba(X_tabular)[:, 1]

    print(f"Shape X_tabular: {X_tabular.shape}")
    print(f"y shape: {y.shape}")  # Should be (82,)
    print(f"y_train_pred shape: {y_train_pred.shape}")  # Should also be (82,)

    train_metrics = calc_metrics(y=y, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba)

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
                  nominal_features, y,
                  imp_max_iter, class_max_iter, pca):
    dataset = dataset_name

    config = DATASET_CONFIGS[dataset]
    n_splits = config.splits
    n_components = config.pca if pca else None
    n_repeats = config.n_repeats

    ml_method = "Logistic Regression"
    emb_method = "Random Trees Embedding"
    concatenation = "yes"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)
    pca_transformer = PCA(n_components=n_components, svd_solver='auto') if pca else "passthrough"
    numerical_features = list(set(X_tabular.columns) - set(nominal_features))

    num_pipeline_steps = [
        ("debug_numerical", DebugTransformer(name="Numerical Debug")),
        ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
        ("debug_numerical_after", DebugTransformer(name="Numerical Debug after"))
    ]
    if pca:
        num_pipeline_steps.append(("scaler", StandardScaler()))

    pipeline = Pipeline([
        ("feature_combiner", FeatureUnion([
            # Verarbeitung der tabellarischen Daten
            ("raw", ColumnTransformer([
                ("nominal", Pipeline([
                    ("debug_nominal", DebugTransformer(name="Nominal Debug")),  # 5
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),  # 14
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
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    )

    for train_index, test_index in skf.split(X_tabular, y):
        X_tab_train, X_tab_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        search.fit(X_tab_train, y_train)

        y_test_pred = search.predict(X_tab_test)
        y_test_pred_proba = search.predict_proba(X_tab_test)[:, 1]

        metrics_per_fold.append(
            calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    search.fit(X_tabular, y)
    y_train_pred = search.predict(X_tabular)
    y_train_pred_proba = search.predict_proba(X_tabular)[:, 1]

    train_metrics = calc_metrics(y=y, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba)

    best_params = f"{search.best_params_}"

    print(f"Feature set size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, best_params, n_components, train_metrics, metrics_per_fold


# läuft
def concat_hgbc_txt_emb(dataset_name, emb_method,
                        X_tabular, y, text_feature_column_name, feature_extractor,
                        nominal_features, raw_text_summaries, concatenation, pca):
    dataset = dataset_name
    config = DATASET_CONFIGS[dataset]
    n_splits = config.splits
    n_components = config.pca if pca else None
    n_repeats = config.n_repeats
    y = pd.Series(y)
    if not np.issubdtype(y.dtype, np.number):
        print(f"Label encoding: {y.unique()}")
        le = LabelEncoder()
        y = le.fit_transform(y)  # this returns a NumPy array
    else:
        y = y.to_numpy()

    ml_method = "HistGradientBoostingClassifier"
    emb_method = emb_method
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)

    # add text as a new column
    text_features = [text_feature_column_name]

    nominal_feature_indices = [X_tabular.columns.get_loc(col) for col in nominal_features]

    X_tabular[text_feature_column_name] = raw_text_summaries

    # separate non-text features
    non_text_columns = list(set(X_tabular.columns) -
                            set(text_features))

    print(f"All columns length: {X_tabular.shape}")
    print(f"Non-text columns length: {len(X_tabular[non_text_columns])}")
    print(f"Non-text columns shape: {X_tabular[non_text_columns].shape}")

    pca_components = f"PCA ({n_components} components)" \
        if pca else "none"

    is_sentence_transformer = False
    if "gtr_t5_base" in emb_method.lower() or "sentence_t5_base" in emb_method.lower() or "modernbert_embed" in emb_method.lower():
        is_sentence_transformer = True

    pipeline_text_steps = [
        ("embedding_aggregator", EmbeddingAggregator(
            feature_extractor=feature_extractor,
            is_sentence_transformer=is_sentence_transformer)),
        # ("debug_text", DebugTransformer(name="Text Debug"))
    ]
    if pca:
        pipeline_text_steps.append(("numerical_scaler", StandardScaler()))
        pipeline_text_steps.append(("pca", PCA(n_components=n_components)))
    else:
        pipeline_text_steps.append(("numerical_scaler", MinMaxScaler()))

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("text", Pipeline(pipeline_text_steps), text_features)
            ])),
            ("classifier", HistGradientBoostingClassifier(categorical_features=nominal_feature_indices))
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
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    )

    # === Evaluation ===
    if dataset_name == DatasetName.POSTTRAUMA.value:
        for train_index, test_index in skf.split(X_tabular, y):
            X_train, X_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print(f"Length of X_tab_train: {len(X_train)}")
            print(f"Length of y_train: {len(y_train)}")

            assert len(X_train) == len(y_train), "Mismatch in training data sizes"

            search.fit(X_train, y_train)

            y_test_pred = search.predict(X_test)
            y_test_pred_proba = search.predict_proba(X_test)[:, 1]

            metrics_per_fold.append(
                calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    elif dataset_name == DatasetName.CYBERSECURITY.value or dataset_name == DatasetName.LUNG_DISEASE.value:
        X_train, X_test, y_train, y_test = train_test_split(X_tabular, y, test_size=0.2, random_state=42)

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Fit and evaluate
        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        best_param = f"Best params for this fold: {search.best_params_}"
        print(best_param)

        metrics_per_fold.append(
            calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

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

    train_metrics = calc_metrics(y=y, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba)

    best_params = f"{search.best_params_}"
    print(f"Best hyperparameters: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics per fold: {metrics_per_fold}")

    return (dataset, ml_method, emb_method, concatenation, best_params,
            pca_components, train_metrics, metrics_per_fold)


def concat_hgbc_rte(dataset_name, X_tabular, y, nominal_features, imp_max_iter, pca):
    dataset = dataset_name
    config = DATASET_CONFIGS[dataset]
    n_splits = config.splits
    n_components = config.pca if pca else None
    n_repeats = config.n_repeats

    ml_method = "HistGradientBoosting"
    emb_method = "Random Trees Embedding"
    concatenation = "yes"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)
    categorical_indices = [X_tabular.columns.get_loc(col) for col in nominal_features]
    numerical_features = list(set(X_tabular.columns) - set(nominal_features))
    pca_transformer = PCA(n_components=n_components, svd_solver='auto') if pca else "passthrough"

    print(f"type of X: {type(X_tabular)}")
    num_pipeline_steps = [
        ("debug_numerical", DebugTransformer(name="Numerical Debug")),
        ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
        ("debug_numerical_after", DebugTransformer(name="Numerical Debug after"))
    ]
    if pca:
        num_pipeline_steps.append(("scaler", StandardScaler()))

    pipeline = Pipeline([
        ("feature_combiner", FeatureUnion([
            ("raw", "passthrough"),
            ("embeddings", Pipeline([
                ("transformer", ColumnTransformer([
                    ("nominal", Pipeline([
                        ("debug_nominal", DebugTransformer(name="Nominal Debug")),
                        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                        ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
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
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    )

    for train_index, test_index in skf.split(X_tabular, y):
        X_train, X_test = X_tabular.iloc[train_index], X_tabular.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        search.fit(X_train, y_train)
        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        metrics_per_fold.append(
            calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba))

    search.fit(X_tabular, y)
    y_train_pred = search.predict(X_tabular)
    y_train_pred_proba = search.predict_proba(X_tabular)[:, 1]

    train_metrics = calc_metrics(y=y, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba)

    best_params = f"{search.best_params_}"

    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Combined train score: {train_metrics}")
    print(f"Combined test scores: {metrics_per_fold}")

    return dataset, ml_method, emb_method, concatenation, best_params, n_components, train_metrics, metrics_per_fold


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
