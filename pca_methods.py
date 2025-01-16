from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, recall_score, precision_score, \
    f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from text_emb_aggregator import EmbeddingAggregator


# Numerical Data: PCA requires numerical data. Categorical data needs to be encoded it first (e.g., one-hot encoding).
# Scaling: Features with different ranges should be scaled (e.g., using StandardScaler or MinMaxScaler) before applying PCA.


def log_reg_pca(dataset_name, X, y, nominal_features, n_splits=3,  n_components=None):
    dataset = dataset_name
    ml_method = "logistic regression"
    emb_method = f"PCA ({n_components} components)" if n_components else "none"
    concatenation = "no"
    metrics_per_fold = []
    skf = StratifiedKFold(n_splits=n_splits)

    pca_step = ("pca", PCA(n_components=n_components)) if n_components else None

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
                ] ), list(set(X.columns.values) - set(nominal_features))),
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

    return dataset, ml_method, emb_method, concatenation, train_metrics, metrics_per_fold

def lr_txt_emb_pca(dataset_name, emb_method, feature_extractor, summaries, y, n_splits=3):
    dataset = dataset_name
    ml_method = "logistic regression"
    concatenation = "no"
    pca = "yes"
    metrics_per_fold = []
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
        cv=RepeatedStratifiedKFold(n_splits=3)
    ) # todo: continue
