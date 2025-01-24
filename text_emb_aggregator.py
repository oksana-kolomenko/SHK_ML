import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class EmbeddingAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractor, method="embedding_cls"):
        self.method = method
        self.feature_extractor = feature_extractor
        self.data_ = None

    # Create embedding based on [CLS] token
    def _embedding_cls(self, text_features):
        print(f"Type of text_features: {type(text_features)}")
        print(f"First element of text_features: {text_features[0]}")
        print(f"Length of text_features: {len(text_features)}")
        embeddings = []
        for summary in text_features:
            embedding = self.feature_extractor(summary)[0][0]
            #print(f"Embedding cls as it is: {embedding}")
            embeddings.append(embedding)
        print(len(embeddings))
        return np.array(embeddings)

    # Create mean embedding excluding [CLS] and [SEP] tokens
    def _embedding_mean_without_cls_and_sep(self, patient_summaries, X):
        embeddings = []
        for summary in patient_summaries:
            embedding = self.feature_extractor(summary)[0][1:-1]
            #print(f"Embedding no_cls_no_sep shape: {np.array(embedding).shape}")
            embeddings.append(np.mean(embedding, axis=0))
        print(len(embeddings))
        return np.array(embeddings)

    # Create mean embedding including [CLS] and [SEP] tokens
    def _embedding_mean_with_cls_and_sep(self, patient_summaries, X):
        embeddings = []
        for summary in patient_summaries:
            embedding = self.feature_extractor(summary)[0][:]
            #print(f"Embedding cls_and_sep shape: {np.array(embedding).shape}")
            embeddings.append(np.mean(embedding, axis=0))
            # todo: add for others
            print("Embedding cls_and_sep dimension" + np.mean(embedding, axis=0))
        print(len(embeddings))
        return np.array(embeddings)

    def fit(self, X, y=None):  # X - summaries
        return self

    def transform(self, X_text):
        print(f"X_text shape:: {X_text.shape}")
        print(f"Input to EmbeddingAggregator: {X_text}")
        if isinstance(X_text, pd.DataFrame):
            X_text = X_text.iloc[:, 0].tolist()
        if self.method == "embedding_cls":
            if not all(isinstance(x, str) for x in X_text):
                raise ValueError("All inputs to EmbeddingAggregator must be strings.")
            return self._embedding_cls(X_text)

        else:
            raise ValueError("Invalid aggregation method")

        """        elif self.method == "embedding_mean_with_cls_and_sep":
            if not all(isinstance(x, str) for x in X_text):
                raise ValueError("All inputs to EmbeddingAggregator must be strings.")
            return self._embedding_mean_with_cls_and_sep(X_text, X)
        elif self.method == "embedding_mean_without_cls_and_sep":
            if not all(isinstance(x, str) for x in X_text):
                raise ValueError("All inputs to EmbeddingAggregator must be strings.")
            return self._embedding_mean_without_cls_and_sep(X_text, X)"""
