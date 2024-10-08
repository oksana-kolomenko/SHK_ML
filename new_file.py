import torch
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModel, pipeline  # AutoModelForMaskedLM
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
model = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
feature_extractor = pipeline(
    "feature-extraction",
    model=model,
    tokenizer=tokenizer
)


# Load features from table
def load_features(file_path = "X.csv", delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter)
    return data.values


# Load features als text (create if doesn't exist)
def load_summaries():
    with open("Summaries.txt", "r") as file:
        summaries_list = [line.strip() for line in file.readlines()]
    return summaries_list


# Create embeddings for the text-summaries
def embedding_with_cls_token(patient_summaries):
    embeddings = feature_extractor(patient_summaries)
    #for summary in patient_summaries:
    #    embeddings.append(feature_extractor(summary))

    emb_with_cls_token = np.array(embeddings[0][0])
    return emb_with_cls_token


def embedding_with_cls_and_sep_tokens(patient_summaries):
    embeddings = []
    for summary in patient_summaries:
        embeddings.append(feature_extractor(summary))

    embedding_mean_with_cls_and_sep_tokens = np.mean(embeddings[0][:], axis=0)
    return embedding_mean_with_cls_and_sep_tokens


def embedding_without_cls_and_sep_tokens(patient_summaries):
    embeddings = []
    for summary in patient_summaries:
        embeddings.append(feature_extractor(summary))

    embedding_mean_without_cls_and_sep_tokens = np.mean(embeddings[0][1:-1], axis=0)
    return embedding_mean_without_cls_and_sep_tokens

"""    model.eval()
    embeddings = []

    with torch.no_grad():
        for summary in patient_summaries:
            inputs = tokenizer(summary, return_tensors='pt', truncation=True,
                               max_length=max_length, padding='max_length')
            outputs = model(**inputs)
            last_hidden_state = outputs[0]

            embedding = last_hidden_state.mean(dim=1).squeeze()
            embeddings.append(embedding.cpu().numpy())
    return np.array(embeddings)"""

# Load labels
def load_labels(file_path = "y.csv", delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter)
    return np.array(data.values.ravel())


"""
1. Logistische Regression ohne Embedding (OHNE TEXT)
train AUC etwa 0.99
test AUC etwa 0.90
"""


def lg_reg(X, y, n_splits=3):
    # split the data
    # n_splits kleiner -> größere AUC?
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    skf.get_n_splits(X, y)
    print(skf)

    lr_model_test_scores = []

    for train_index, test_index in skf.split(X, y):
        # split data
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

        # No Embedding
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, y_train)

        lr_model_test_scores.append(roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1]))

    # impute data
    imputer = KNNImputer()
    X = imputer.fit_transform(X)

    # Scale data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X, y)

    # Get results
    lr_model_train_score = roc_auc_score(y, lr_model.predict_proba(X)[:, 1])

    print(f"LR_model train Scores: {lr_model_train_score}")
    print(f"LR_model test Scores: {lr_model_test_scores}")

    return lr_model_train_score, lr_model_test_scores


"""
2. Logistische Regression mit RandomTree-Embedding (OHNE TEXT)
"""


def lg_reg_emb(X, y, n_splits=3, n_estimators=10, max_depth=3):
    # split the data
    # n_splits kleiner -> größere AUC?
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    skf.get_n_splits(X, y)
    print(skf)

    rt_model_test_scores = []

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
            n_estimators=n_estimators, random_state=42, max_depth=max_depth)  # FIND USEFULL PARAMS
        rt_model = make_pipeline(random_tree_embedding, LogisticRegression(max_iter=1000))
        rt_model.fit(X_train, y_train)

        rt_model_test_scores.append(roc_auc_score(y_test, rt_model.predict_proba(X_test)[:, 1]))

    imputer = KNNImputer()
    X = imputer.fit_transform(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    random_tree_embedding = RandomTreesEmbedding(
        n_estimators=n_estimators, random_state=42, max_depth=max_depth)  # find useful params
    rt_model = make_pipeline(random_tree_embedding, LogisticRegression(max_iter=1000))
    rt_model.fit(X, y)

    rt_model_train_score = roc_auc_score(y, rt_model.predict_proba(X)[:, 1])

    print(f"RT_model train Scores: {rt_model_train_score}")
    print(f"RT_model test Scores: {rt_model_test_scores}")

    return rt_model_train_score, rt_model_test_scores


"""
3. Logistische Regression mit ClinicalLongformer-Embedding (MIT TEXT)
"""


def lg_re_txt_emb(embeddings, y):
    # erster Wert zu groß
    print(f"Embedding size: {embeddings.shape}")

    # X_train, X_test, y_train, y_test = train_test_split(X_embedded, y, test_size=0.4, random_state=42)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # skf.get_n_splits(X_embedded, y)
    # print(skf)

    test_scores = []

    for train_index, test_index in skf.split(embeddings, y):
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Scale data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, y_train)

        test_scores.append(roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1]))

    # Scale data
    scaler = MinMaxScaler()
    X_embedded = scaler.fit_transform(embeddings)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_embedded, y)

    lr_model_train_score = roc_auc_score(y, lr_model.predict_proba(X_embedded)[:, 1])
    print(f"LR_model train Scores(text): {lr_model_train_score}")
    print(f"LR_model test Scores(text): {test_scores}")

    return lr_model_train_score, test_scores

