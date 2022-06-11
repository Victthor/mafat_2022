
import numpy as np
# from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, min_length=360):
        self.min_length = min_length

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        X_train = [interval[['RSSI_Left', 'RSSI_Right']].to_numpy().transpose() for interval in x]
        X_train = [vec - np.median(vec, axis=1, keepdims=True) for vec in X_train]

        diffs = [(data[0, :] - data[1, :]).astype(np.float32).reshape((1, self.min_length)) for data in X_train]
        avgs = [(data[0, :] + data[1, :]).astype(np.float32).reshape((1, self.min_length)) / 2.0 for data in X_train]
        diffs = [vec - np.mean(vec, axis=1, keepdims=True) for vec in diffs]
        avgs = [vec - np.mean(vec, axis=1, keepdims=True) for vec in avgs]

        # X_train = [np.concatenate([data, diff, avg], axis=0) for data, diff, avg in zip(X_train, diffs, avgs)]
        inputs = [np.concatenate([diff, avg], axis=0) for diff, avg in zip(diffs, avgs)]
        # X_train = avgs

        return np.array(inputs)
