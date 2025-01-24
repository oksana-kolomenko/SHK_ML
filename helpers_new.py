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

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


# Debugging transformer to print array shape
class DebugTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, name="Step"):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"{self.name}: Output shape {np.array(X).shape}")
        return X


def conc_lr_txt_emb_(feature_extractor, raw_text_summaries, X, y, nominal_features, text_feature, imp_max_iter,
                    lr_max_iter, n_repeats, n_splits=3, n_components=10):

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    numerical_features = list(set(X.columns) - set(nominal_features))


    print(f"Tabelle Größe {len(X.columns)}")
    print(f"Tabelle Größe {X.shape}")

    pca_components = f"PCA ({n_components} components)" if n_components else "none"
    pca_step = ("pca", PCA(n_components=n_components)) if n_components else None

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", FeatureUnion([

            ])),
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=lr_max_iter))
        ]),


        param_grid={"classifier__C": [2, 10], #, 50, 250]},
                    "transformer__text__embedding_aggregator__method": [
                        "embedding_cls"
                    ]
                },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3),
        error_score="raise"
    )

    search.fit(X, y)


def conc_lr_txt_emb(feature_extractor, raw_text_summaries, X, y, nominal_features, text_feature, imp_max_iter,
                    lr_max_iter, n_repeats, n_splits=3, n_components=10):

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    # add text as a new column
    text_features = [text_feature]
    X[text_feature] = raw_text_summaries
    numerical_features = list(set(X.columns) - set(nominal_features) - set(text_features))

    assert len(X[text_feature]) == 82, "Text features are more than 82"

    print(f"Tabelle Größe {len(X.columns)}")
    print(f"Tabelle Größe {X.shape}")

    print(f"X.columns: {X.columns}")
    print(f"Type text features: {type(text_features)}")
    print("Text features:", text_features)

    pca_components = f"PCA ({n_components} components)" if n_components else "none"
    pca_step = ("pca", PCA(n_components=n_components)) if n_components else None

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("debug_nominal", DebugTransformer(name="Nominal Debug")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), nominal_features),  # these are columns only
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
                    ("debug_numerical", DebugTransformer(name="Numerical Debug")),
                    ("numerical_scaler", MinMaxScaler())
                ]), numerical_features),
                ("text", Pipeline([
                    ("embedding_aggregator", EmbeddingAggregator(feature_extractor)),
                    ("debug_text", DebugTransformer(name="Text Debug")),
                    ("numerical_scaler", MinMaxScaler()),
                ]), text_features), # todo: as columnname übergeben, im Emb. Aggr. Eingabetype anpassen und dort die Tabelle haben
                # + ([pca_step] if pca_step else [])), X[text_column_name].tolist()),  # todo: text is one col pro row, after embedding
                # todo: should be 768 and after pca 40!
            ])),
            ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=lr_max_iter))
        ]),
        param_grid={"classifier__C": [2, 10], #, 50, 250]},
                    "transformer__text__embedding_aggregator__method": [
                        "embedding_cls"
                    ]
                },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=3),
        error_score="raise"
    )

    search.fit(X, y)

    y_pred = search.predict(X)
    y_pred_proba = search.predict_proba(X)[:1]

    print(f"Y pred: {y_pred}")
    print(f"Y pred proba: {y_pred_proba}")
