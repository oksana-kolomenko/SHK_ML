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
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from csv_parser import create_patient_summaries

identifier = "yikuan8/Clinical-Longformer"
tokenizer = AutoTokenizer.from_pretrained(identifier)
model = AutoModel.from_pretrained(identifier)
feature_extractor = pipeline(
    "feature-extraction",
    model=model,
    tokenizer=tokenizer
)


# Load features
def load_features(file_path="X.csv", delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter)
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


"""class HGBClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, nominal_features, min_samples_leaf=10):
        self.nominal_features = nominal_features
        self.min_samples_leaf = min_samples_leaf
        self.model = HistGradientBoostingClassifier(
            categorical_features=self.nominal_features,
            min_samples_leaf=self.min_samples_leaf
        )

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)
"""


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

class HGBClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, nominal_features=None, min_samples_leaf=10):
        self.nominal_features = nominal_features if nominal_features is not None else []
        self.min_samples_leaf = min_samples_leaf
        self.model = HistGradientBoostingClassifier(min_samples_leaf=self.min_samples_leaf)
        self.encoder = None if not self.nominal_features else OrdinalEncoder()

    def fit(self, X, y=None):
        # If nominal features are specified, transform them
        if self.nominal_features:
            # Fit the encoder only on nominal columns
            X = X.copy()  # To avoid modifying the original data
            X[self.nominal_features] = self.encoder.fit_transform(X[self.nominal_features])
        # Fit the model on the transformed data
        self.model.fit(X, y)
        return self

    def transform(self, X):
        # Transform the nominal features for test or unseen data
        if self.nominal_features:
            X = X.copy()
            X[self.nominal_features] = self.encoder.transform(X[self.nominal_features])
        # Return the probability of the positive class as a single-column output
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)


class EmbeddingAggregator(BaseEstimator, TransformerMixin):
    # Aggregator-Klasse für die Pipeline
    def __init__(self, method="embedding_cls"):
        self.method = method

    def fit(self, X, y=None):  # X - summaries
        return self

    def transform(self, X):
        if self.method == "embedding_cls":
            return self._embedding_cls(X)
        elif self.method == "embedding_mean_with_cls_and_sep":
            return self._embedding_mean_with_cls_and_sep(X)
        elif self.method == "embedding_mean_without_cls_and_sep":
            return self._embedding_mean_without_cls_and_sep(X)
        else:
            raise ValueError("Invalid aggregation method")

    # Create embedding based on [CLS] token
    def _embedding_cls(self, patient_summaries):
        embeddings = []
        for summary in patient_summaries:
            embeddings.append(feature_extractor(summary)[0][0])
        return np.array(embeddings)

    # Create mean embedding including [CLS] and [SEP] tokens
    def _embedding_mean_with_cls_and_sep(self, patient_summaries):
        embeddings = []
        for summary in patient_summaries:
            embeddings.append(np.mean(feature_extractor(summary)[0][:], axis=0))
        return np.array(embeddings)

    # Create mean embedding excluding [CLS] and [SEP] tokens
    def _embedding_mean_without_cls_and_sep(self, patient_summaries):
        embeddings = []
        for summary in patient_summaries:
            embeddings.append(np.mean(feature_extractor(summary)[0][1:-1], axis=0))
        return np.array(embeddings)


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


def lr_txt_emb(embeddings, y, n_splits=3):  # todo
    lr_txt_emb_test_scores = []
    skf = StratifiedKFold(n_splits=n_splits)

    search = GridSearchCV(
        estimator=Pipeline([
            ("numerical_scaler", MinMaxScaler()),
            ("aggregator", EmbeddingAggregator()),
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

    for train_index, test_index in skf.split(embeddings, y):
        X_train, X_test = [embeddings[i] for i in train_index], [embeddings[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit and evaluate
        search.fit(X_train, y_train)
        y_pred_proba = search.predict_proba(X_test)[:, 1]
        lr_txt_emb_test_scores.append(roc_auc_score(y_test, y_pred_proba))

    search.fit(
        embeddings,
        y
    )

    lr_txt_emb_train_score = roc_auc_score(y, search.predict_proba(embeddings)[:, 1])
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
                ("hist_gb", HGBClassifierWrapper(nominal_features))
            ]),
            param_grid={"hist_gb__min_samples_leaf": [5, 10, 15, 20]
                        },
            scoring="neg_log_loss",
            cv=3,
            refit="hist_gb"
        )

        search.fit(
            X_train,
            y_train
        )
        hgbc_test_scores.append(roc_auc_score(y_test, search.predict_log_proba(X_test)[:, 1]))

    # HGB on whole data (test)
    search = GridSearchCV(
        estimator=Pipeline([
            ("hist_gb", HGBClassifierWrapper(nominal_features))
        ]),
        param_grid={"hist_gb__min_samples_leaf": [5, 10, 15, 20]
                    },
        scoring="neg_log_loss",
        cv=3,
        refit="hist_gb"
    )
    search.fit(
        X,
        y
    )
    hgbc_train_score = roc_auc_score(y, search.predict_log_proba(X)[:, 1])

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
                ("hist_gb", HistGradientBoostingClassifier(categorical_features=nominal_features))
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
            ("hist_gb", HistGradientBoostingClassifier(categorical_features=nominal_features))
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


def hgbc_txt_emb(embeddings, nominal_features, y, n_splits=3):
    hgbc_txt_emb_test_scores = []
    skf = StratifiedKFold(n_splits=n_splits)

    for train_index, test_index in skf.split(embeddings, y):
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # todo: scalieren

        # search for test scores
        search = GridSearchCV(
            estimator=Pipeline([
                ("numerical_scaler", MinMaxScaler()),
                ("hist_gb", HGBClassifierWrapper(nominal_features)),
                ("aggregator", EmbeddingAggregator())
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
            X_train,
            y_train
        )
        hgbc_txt_emb_test_scores.append(roc_auc_score(y_test, search.predict_proba(X_test)[:, 1]))

    # search for train scores
    search = GridSearchCV(
        estimator=Pipeline([
            ("numerical_scaler", MinMaxScaler()),
            ("hist_gb", HGBClassifierWrapper(nominal_features)),
            ("aggregator", EmbeddingAggregator())
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
        embeddings,
        y
    )

    hgbc_txt_emb_train_score = roc_auc_score(y, search.predict_proba(embeddings)[:, 1])
    print(f"embedding size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"best hyperparameters: {search.best_params_}")
    print(f"hgbc_txt_emb_train_score: {hgbc_txt_emb_train_score}")
    print(f"hgbc_txt_emb_test_scores: {hgbc_txt_emb_test_scores}")

    return hgbc_txt_emb_train_score, hgbc_txt_emb_test_scores
