from sklearn.base import BaseEstimator, TransformerMixin

class BinaryEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.map(lambda x: 1 if x == 'yes' else 0)