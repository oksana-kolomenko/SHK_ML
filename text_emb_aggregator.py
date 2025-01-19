from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class EmbeddingAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractor, method="embedding_cls"):
        self.method = method
        self.feature_extractor = feature_extractor

    # Create mean embedding including [CLS] and [SEP] tokens
    def _embedding_mean_with_cls_and_sep(self, patient_summaries):
        embeddings = []
        for summary in patient_summaries:
            embedding = self.feature_extractor(summary)[0][:]
            print(f"Embedding cls_and_sep shape: {np.array(embedding).shape}")
            embeddings.append(np.mean(embedding, axis=0))
        print(len(embeddings))
        return np.array(embeddings)

    # Create embedding based on [CLS] token
    def _embedding_cls(self, patient_summaries):
        embeddings = []
        for summary in patient_summaries:
            embedding = self.feature_extractor(summary)[0][0]
            print(f"Embedding cls shape: {np.array(embedding).shape}")
            embeddings.append(embedding)
        print(len(embeddings))
        return np.array(embeddings)

    # Create mean embedding excluding [CLS] and [SEP] tokens
    def _embedding_mean_without_cls_and_sep(self, patient_summaries):
        embeddings = []
        for summary in patient_summaries:
            embedding = self.feature_extractor(summary)[0][1:-1]
            print(f"Embedding no_cls_no_sep shape: {np.array(embedding).shape}")
            embeddings.append(np.mean(embedding, axis=0))
        print(len(embeddings))
        return np.array(embeddings)

    def fit(self, X, y=None):  # X - summaries
        return self

    def transform(self, X):
        if self.method == "embedding_cls":
            if not all(isinstance(x, str) for x in X):
                raise ValueError("All inputs to EmbeddingAggregator must be strings.")
            return self._embedding_cls(X)
        elif self.method == "embedding_mean_with_cls_and_sep":
            if not all(isinstance(x, str) for x in X):
                raise ValueError("All inputs to EmbeddingAggregator must be strings.")
            return self._embedding_mean_with_cls_and_sep(X)
        elif self.method == "embedding_mean_without_cls_and_sep":
            if not all(isinstance(x, str) for x in X):
                raise ValueError("All inputs to EmbeddingAggregator must be strings.")
            return self._embedding_mean_without_cls_and_sep(X)
        else:
            raise ValueError("Invalid aggregation method")
