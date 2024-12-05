import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomTreesEmbedding, HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.base import TransformerMixin, BaseEstimator

from csv_parser import create_patient_summaries

from transformers import AutoTokenizer, AutoModel, pipeline

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
        return create_patient_summaries()
    with open("Summaries.txt", "r") as file:
        summaries_list = [line.strip() for line in file.readlines()]
    return summaries_list


def logistic_regression(X, y, nominal_features, n_splits=3):
    lg_reg_test_scores = []
    skf = StratifiedKFold(n_splits=n_splits)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # search for test scores
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
                    ]), list(set(X_train.columns.values) - set(nominal_features))),
                ])),
                ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
            ]),
            param_grid={"classifier__C": [2, 10, 50, 250]},
            scoring="neg_log_loss",
            cv=3
        )

        search.fit(
            X_train,
            y_train
        )

        lg_reg_test_scores.append(roc_auc_score(y_test, search.predict_proba(X_test)[:, 1]))

    # search for train scores
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
                ]), list(set(X_train.columns.values) - set(nominal_features))),
            ])),
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
        ]),
        param_grid={"classifier__C": [2, 10, 50, 250]},
        scoring="neg_log_loss",
        cv=3
    )

    search.fit(
        X,
        y
    )

    lg_reg_train_score = roc_auc_score(y, search.predict_proba(X)[:, 1])
    print(f"feature set size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"best hyperparameters: {search.best_params_}")
    print(f"lg_reg_train_score: {lg_reg_train_score}")
    print(f"lg_reg_test_scores: {lg_reg_test_scores}")

    return lg_reg_train_score, lg_reg_test_scores


def lr_ran_tree_emb(X, y, nominal_features, n_splits=3):
    lg_reg_rt_emb_test_scores = []
    skf = StratifiedKFold(n_splits=n_splits)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # search for test scores
        search = GridSearchCV(
            estimator=Pipeline([
                ("transformer", ColumnTransformer([
                    ("nominal", Pipeline([
                        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                        ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                    ]), nominal_features),
                    ("numerical", Pipeline([
                        ("numerical_imputer", IterativeImputer(max_iter=30))
                    ]), list(set(X_train.columns.values) - set(nominal_features))),
                ])),
                ("embedding", RandomTreesEmbedding()),
                ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
            ]),
            param_grid={
                # extended hyperparameter search for better embedding
                "embedding__n_estimators": [10, 100, 1000],
                "embedding__max_depth": [2, 5, 10, 15],
                "classifier__C": [2, 10, 50, 250]
            },
            scoring="neg_log_loss",
            cv=3
        )

        search.fit(
            X_train,
            y_train
        )

        lg_reg_rt_emb_test_scores.append(roc_auc_score(y_test, search.predict_proba(X_test)[:, 1]))

    # search for train scores
    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), nominal_features),
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=30))
                ]), list(set(X_train.columns.values) - set(nominal_features))),
            ])),
            ("embedding", RandomTreesEmbedding()),
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
        ]),
        param_grid={
            # extended hyperparameter search for better embedding
            "embedding__n_estimators": [10, 100, 1000],
            "embedding__max_depth": [2, 5, 10, 15],
            "classifier__C": [2, 10, 50, 250]
        },
        scoring="neg_log_loss",
        cv=3
    )

    search.fit(
        X,
        y
    )

    lg_reg_rt_emb_train_score = roc_auc_score(y, search.predict_proba(X)[:, 1])
    print(f"embedding size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"best hyperparameters: {search.best_params_}")
    print(f"lg_reg_rt_emb_train_score: {lg_reg_rt_emb_train_score}")
    print(f"lg_reg_rt_emb_test_scores: {lg_reg_rt_emb_test_scores}")

    return lg_reg_rt_emb_train_score, lg_reg_rt_emb_test_scores


def lr_txt_emb(feature_extractor, summaries, y, n_splits=3):  # todo
    lr_txt_emb_test_scores = []
    skf = StratifiedKFold(n_splits=n_splits)

    search = GridSearchCV(
        estimator=Pipeline([

            ("aggregator", EmbeddingAggregator(feature_extractor)),
            ("numerical_scaler", MinMaxScaler()),
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
        cv=3
    )
    for train_index, test_index in skf.split(summaries, y):
        X_train, X_test = [summaries[i] for i in train_index], [summaries[i] for i in test_index]
        #X_train, X_test = summaries[train_index], summaries[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #X_train, X_test = np.array(X_train), np.array(X_test)  # need?

        # Fit and evaluate
        search.fit(X_train, y_train)
        y_pred_proba = search.predict_proba(X_test)[:, 1]
        lr_txt_emb_test_scores.append(roc_auc_score(y_test, y_pred_proba))

    search.fit(
        summaries,
        y
    )

    lr_txt_emb_train_score = roc_auc_score(y, search.predict_proba(summaries)[:, 1])
    print(f"embedding size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"best hyperparameters: {search.best_params_}")
    print(f"lg_reg_txt_emb_train_score: {lr_txt_emb_train_score}")
    print(f"lg_reg_txt_emb_test_scores: {lr_txt_emb_test_scores}")

    return lr_txt_emb_train_score, lr_txt_emb_test_scores


def hgbc(X, y, nominal_features, n_splits=3):
    hgbc_test_scores = []
    skf = StratifiedKFold(n_splits=n_splits)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # search for test scores
        search = GridSearchCV(
            estimator=Pipeline([
                ("hist_gb", HistGradientBoostingClassifier(categorical_features=nominal_features))
            ]),
            param_grid={"hist_gb__min_samples_leaf": [5, 10, 15, 20]
                        },
            scoring="neg_log_loss",
            cv=3
        )

        search.fit(
            X_train,
            y_train
        )
        hgbc_test_scores.append(roc_auc_score(y_test, search.predict_proba(X_test)[:, 1]))

    # HGB on whole data (test)
    search = GridSearchCV(
        estimator=Pipeline([
            ("hist_gb", HistGradientBoostingClassifier(categorical_features=nominal_features))
        ]),
        param_grid={"hist_gb__min_samples_leaf": [5, 10, 15, 20]
                    },
        scoring="neg_log_loss",
        cv=3
    )
    search.fit(
        X,
        y
    )
    hgbc_train_score = roc_auc_score(y, search.predict_proba(X)[:, 1])

    print(f"best hyperparameters: {search.best_params_}")
    print(f"hgbc_train_score: {hgbc_train_score}")
    print(f"hgbc_test_scores: {hgbc_test_scores}")

    return hgbc_train_score, hgbc_test_scores


def hgbc_ran_tree_emb(X, y, nominal_features, n_splits=3):  # todo
    hgbc_rt_emb_test_scores = []
    skf = StratifiedKFold(n_splits=n_splits)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # search for test scores
        search = GridSearchCV(
            estimator=Pipeline([
                ("transformer", ColumnTransformer([
                    ("nominal", Pipeline([
                        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                        ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                    ]), nominal_features),
                    ("numerical", Pipeline([
                        ("numerical_imputer", IterativeImputer(max_iter=30))
                    ]), list(set(X_train.columns.values) - set(nominal_features))),
                ])),
                ("embedding", RandomTreesEmbedding()),
                ("hist_gb", HistGradientBoostingClassifier())
            ]),
            param_grid={
                # extended hyperparameter search for better embedding
                "embedding__n_estimators": [10, 100, 1000],
                "embedding__max_depth": [2, 5, 10, 15],
                "hist_gb__min_samples_leaf": [5, 10, 15, 20]
            },
            scoring="neg_log_loss",
            cv=3
        )

        search.fit(
            X_train,
            y_train
        )

        hgbc_rt_emb_test_scores.append(roc_auc_score(y_test, search.predict_proba(X_test)[:, 1]))

    # search for train scores
    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), nominal_features),
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=30))
                ]), list(set(X_train.columns.values) - set(nominal_features))),
            ])),
            ("embedding", RandomTreesEmbedding()),
            ("hist_gb", HistGradientBoostingClassifier())
        ]),
        param_grid={
            # extended hyperparameter search for better embedding
            "embedding__n_estimators": [10, 100, 1000],
            # best token output with name of the hyperparameter
            "embedding__max_depth": [2, 5, 10, 15],
            "hist_gb__min_samples_leaf": [5, 10, 15, 20]
        },
        scoring="neg_log_loss",
        cv=3
    )

    search.fit(
        X,
        y
    )

    hgbc_rt_emb_train_score = roc_auc_score(y, search.predict_proba(X)[:, 1])
    print(f"embedding size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"best hyperparameters: {search.best_params_}")
    print(f"lr_ran_tree_emb_train_score: {hgbc_rt_emb_train_score}")
    print(f"lr_ran_tree_emb_test_scores: {hgbc_rt_emb_test_scores}")

    return hgbc_rt_emb_train_score, hgbc_rt_emb_test_scores


def hgbc_txt_emb(feature_extractor, summaries, nominal_features, y, n_splits=3):
    hgbc_txt_emb_test_scores = []
    skf = StratifiedKFold(n_splits=n_splits)

    for train_index, test_index in skf.split(summaries, y):
        #X_train, X_test = summaries[train_index], summaries[test_index]
        X_train, X_test = [summaries[i] for i in train_index], [summaries[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_test = np.array(X_train), np.array(X_test)

        # search for test scores
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
            cv=3
        )
        search.fit(
            np.array(X_train),
            y_train
        )
        hgbc_txt_emb_test_scores.append(roc_auc_score(y_test, search.predict_proba(X_test)[:, 1]))

    # search for train scores
    search = GridSearchCV(
        estimator=Pipeline([
            ("aggregator", EmbeddingAggregator(feature_extractor)), #  todo: macht es Sinn mit anderen Sprachmodellen?
            # todo: 1) mit der transf. Funktion versuchen,
            # todo: 2) test if numerical value comes out
            ("numerical_scaler", MinMaxScaler()),
            ("hist_gb", HistGradientBoostingClassifier()),

        ]),
        param_grid={
            "hist_gb__min_samples_leaf": [5, 10, 15, 20],
            "aggregator__method": ["embedding_cls", "embedding_mean_with_cls_and_sep",
                                   "embedding_mean_without_cls_and_sep"]
        },
        scoring="neg_log_loss",
        cv=3
    )

    search.fit(
        np.array(summaries),
        y
    )

    hgbc_txt_emb_train_score = roc_auc_score(y, search.predict_proba(np.array(summaries))[:, 1])
    print(f"best aggregator: {len(search.best_estimator_.named_steps['aggregator'].coef_[0])}")
    print(f"best hyperparameters: {search.best_params_}")
    print(f"hgbc_txt_emb_train_score: {hgbc_txt_emb_train_score}")
    print(f"hgbc_txt_emb_test_scores: {hgbc_txt_emb_test_scores}")

    return hgbc_txt_emb_train_score, hgbc_txt_emb_test_scores
