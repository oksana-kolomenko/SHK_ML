import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

import new_file
from bar_plotting import plot_bar_chart


def draw_curve():
    # load features
    X = new_file.load_features("../X.csv", delimiter=',')
    y = new_file.load_labels("../y.csv")

    # split the data
    # n_splits kleiner -> größere AUC?
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    skf.get_n_splits(X, y)
    print(skf)

    rt_model_test_scores = []
    lr_model_test_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Replace NaNs
        imputer = KNNImputer()
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Scale data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Embedding
        random_tree_embedding = RandomTreesEmbedding(
            n_estimators=10, random_state=42, max_depth=3)  # FIND USEFULL PARAMS
        rt_model = make_pipeline(random_tree_embedding, LogisticRegression(max_iter=1000))
        rt_model.fit(X_train, y_train)

        # No Embedding
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, y_train)

        rt_model_test_scores.append(roc_auc_score(y_test, rt_model.predict_proba(X_test)[:, 1]))
        lr_model_test_scores.append(roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1]))

    # impute data
    imputer = KNNImputer()
    X = imputer.fit_transform(X)

    # Scale data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # LR mit embedding
    random_tree_embedding = RandomTreesEmbedding(
        n_estimators=10, random_state=42, max_depth=3)  # find useful params
    rt_model = make_pipeline(random_tree_embedding, LogisticRegression(max_iter=1000))
    rt_model.fit(X, y)

    # LR ohne embedding
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X, y)

    # Get results
    rt_model_train_score = roc_auc_score(y, rt_model.predict_proba(X)[:, 1])
    lr_model_train_score = roc_auc_score(y, lr_model.predict_proba(X)[:, 1])

    print(f"RT_model train Scores: {rt_model_train_score}")
    print(f"RT_model test Scores: {rt_model_test_scores}")
    print(f"LR_model train Scores: {lr_model_train_score}")
    print(f"LR_model test Scores: {lr_model_test_scores}")

    #####################
    ### plot the bars ###
    #####################
    labels = ['With embedding', 'Without embedding']
    train_scores = [rt_model_train_score, lr_model_train_score]
    test_means = [np.mean(rt_model_test_scores), np.mean(lr_model_test_scores)]
    test_stds = [np.std(rt_model_test_scores), np.std(lr_model_test_scores)]

    plot_bar_chart(labels, train_scores, test_means, test_stds)
