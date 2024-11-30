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
            embeddings.append(np.mean(self.feature_extractor(summary)[0][:], axis=0))
        return np.array(embeddings)

    # Create embedding based on [CLS] token
    def _embedding_cls(self, patient_summaries):
        embeddings = []
        for summary in patient_summaries:
            embeddings.append(self.feature_extractor(summary)[0][0])
        return np.array(embeddings)

    # Create mean embedding excluding [CLS] and [SEP] tokens
    def _embedding_mean_without_cls_and_sep(self, patient_summaries):
        embeddings = []
        for summary in patient_summaries:
            embeddings.append(np.mean(self.feature_extractor(summary)[0][1:-1], axis=0))
        return np.array(embeddings)

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
