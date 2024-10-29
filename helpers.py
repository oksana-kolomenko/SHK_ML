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
def load_features(file_path = "X.csv", delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter)
    return data


# Load labels
def load_labels(file_path = "y.csv", delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter)
    return np.array(data.values.ravel())


# Load features as text summaries (create if doesn't exist)
def load_summaries():
    if not os.path.exists("Summaries.txt"):
        return create_patient_summaries()
    with open("Summaries.txt", "r") as file:
        summaries_list = [line.strip() for line in file.readlines()]
    return summaries_list


# Create embedding based on [CLS] token
def embedding_cls(patient_summaries):
    embeddings = []
    for summary in patient_summaries:
        embeddings.append(feature_extractor(summary)[0][0])

    return np.array(embeddings)


# Create mean embedding including [CLS] and [SEP] tokens
def embedding_mean_with_cls_and_sep(patient_summaries):
    embeddings = []
    for summary in patient_summaries:
        embeddings.append(np.mean(feature_extractor(summary)[0][:], axis=0))

    return np.array(embeddings)


# Create mean embedding excluding [CLS] and [SEP] tokens
def embedding_mean_without_cls_and_sep(patient_summaries):
    embeddings = []
    for summary in patient_summaries:
        embeddings.append(np.mean(feature_extractor(summary)[0][1:-1], axis=0))

    return np.array(embeddings)


class HGBClassifierWrapper(BaseEstimator, TransformerMixin):
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


class EmbeddingAggregator(BaseEstimator, TransformerMixin):
    # Aggregator-Klasse für die Pipeline
    def __init__(self, method="embedding_cls"):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.method == "embedding_cls":
            return self._embedding_cls(patient_summaries=create_patient_summaries())
        elif self.method == "embedding_mean_with_cls_and_sep":
            return self._embedding_mean_with_cls_and_sep(patient_summaries=create_patient_summaries())
        elif self.method == "embedding_mean_without_cls_and_sep":
            return self._embedding_mean_without_cls_and_sep(patient_summaries=create_patient_summaries())
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


def lg_reg(X, y, nominal_features, n_splits=3):
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


def lr_hgbc(X, y, nominal_features, n_splits=3):
    lg_reg_test_scores = []
    skf = StratifiedKFold(n_splits=n_splits)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # + GridSearch für Parameter
        clf = HistGradientBoostingClassifier(categorical_features=nominal_features, min_samples_leaf=15)
        clf.fit(X_train, y_train)

        lg_reg_test_scores.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    # HGB on whole data (test)
    clf = HistGradientBoostingClassifier(categorical_features=nominal_features, min_samples_leaf=15)
    clf.fit(X, y)

    lg_reg_train_score = roc_auc_score(y, clf.predict_proba(X)[:, 1])

    print(f"lg_reg_hgbc_train_score: {lg_reg_train_score}")
    print(f"lg_reg_hgbc_test_scores: {lg_reg_test_scores}")

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
                "embedding__max_depth": [2, 5, 10],
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
            "embedding__max_depth": [2, 5, 10],
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


def hgbc_ran_tree_emb(X, y, nominal_features, n_splits=3):
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
                ("hist_gb", HistGradientBoostingClassifier(categorical_features=nominal_features))
            ]),
            param_grid={
                # extended hyperparameter search for better embedding
                "embedding__n_estimators": [10, 100, 1000],
                "embedding__max_depth": [2, 5, 10],
                "hist_gb__min_samples_leaf": [10, 15, 20]
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
            ("hist_gb", HistGradientBoostingClassifier(categorical_features=nominal_features))
        ]),
        param_grid={
            # extended hyperparameter search for better embedding
            "embedding__n_estimators": [10, 100, 1000],
            # best token output with name of the hyperparameter
            "embedding__max_depth": [2, 5, 10],
            "hist_gb__min_samples_leaf": [10, 15, 20]
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
    print(f"lr_ran_tree_emb_train_score: {lg_reg_rt_emb_train_score}")
    print(f"lr_ran_tree_emb_test_scores: {lg_reg_rt_emb_test_scores}")

    return lg_reg_rt_emb_train_score, lg_reg_rt_emb_test_scores


def lr_txt_emb(embeddings, y, n_splits=3):
    lg_reg_txt_emb_test_scores = []
    skf = StratifiedKFold(n_splits=n_splits)

    for train_index, test_index in skf.split(embeddings, y):
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # search for test scores
        search = GridSearchCV(
            estimator=Pipeline([
                ("numerical_scaler", MinMaxScaler()),
                ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000)),
                ("aggregator", EmbeddingAggregator()) # todo
            ]),
            param_grid={"classifier__C": [2, 10, 50, 250],
                        # 3 tokens as HP
                        },
            scoring="neg_log_loss",
            cv=3
        )

        search.fit(
            X_train,
            y_train
        )

        lg_reg_txt_emb_test_scores.append(roc_auc_score(y_test, search.predict_proba(X_test)[:, 1]))
    
    # search for train scores
    search = GridSearchCV(
        estimator=Pipeline([
            ("numerical_scaler", MinMaxScaler()),
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
        ]),
        param_grid={"classifier__C": [2, 10, 50, 250]
                    # 3 tokens as HP
                    },
        scoring="neg_log_loss",
        cv=3
    )

    search.fit(
        embeddings,
        y
    )

    lg_reg_txt_emb_train_score = roc_auc_score(y, search.predict_proba(embeddings)[:, 1])
    print(f"embedding size: {len(search.best_estimator_.named_steps['classifier'].coef_[0])}")
    print(f"best hyperparameters: {search.best_params_}")
    print(f"lg_reg_txt_emb_train_score: {lg_reg_txt_emb_train_score}")
    print(f"lg_reg_txt_emb_test_scores: {lg_reg_txt_emb_test_scores}")

    return lg_reg_txt_emb_train_score, lg_reg_txt_emb_test_scores


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
                ("hist_gb", HGBClassifierWrapper(nominal_features))
            ]),
            param_grid={"hist_gb__min_samples_leaf": [5, 10, 15, 20]},
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
            ("hist_gb", HGBClassifierWrapper(nominal_features))
            # todo: 3 tokens
        ]),
        param_grid={"hist_gb__min_samples_leaf": [5, 10, 15, 20]
                    # todo: 3 tokens
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
