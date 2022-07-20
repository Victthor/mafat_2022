
import numpy as np
# from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, params):
        self.params = params
        self.with_mean = params.get('with_mean', 'mean')
        self.with_std = params.get('with_std', False)
        self.feature_set = params.get('feature_set', {'diff': {'apply': True, 'with_mean': 'mean', 'with_std': True}})

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):

        if isinstance(x, list):
            x_train = np.array(
                [interval[['RSSI_Left', 'RSSI_Right']].to_numpy().astype(np.float32).transpose() for interval in x]
            )
        else:
            x_train = x[['RSSI_Left', 'RSSI_Right']].to_numpy().astype(np.float32).transpose()

        if self.with_mean == 'mean':
            x_train -= np.mean(x_train, axis=2, keepdims=True)
        elif self.with_mean == 'median':
            x_train -= np.median(x_train, axis=2, keepdims=True)

        if self.with_std:
            x_train /= (np.std(x_train, axis=2, keepdims=True) + 1e-6)

        feat_list = []

        for feature, fparams in self.feature_set.items():
            if fparams[0]:
                if feature == 'diff':
                    feat_list.append(x_train[:, :1, :] - x_train[:, 1:2, :])
                elif feature == 'avg':
                    feat_list.append(x_train[:, :1, :] + x_train[:, 1:2, :] / 2.0)
                elif feature == 'rssi_left':
                    feat_list.append(x_train[:, :1, :])
                elif feature == 'rssi_right':
                    feat_list.append(x_train[:, 1:2, :])

                if fparams[1] == 'mean':
                    feat_list[-1] -= np.mean(feat_list[-1], axis=2, keepdims=True)
                elif fparams[1] == 'median':
                    feat_list[-1] -= np.median(feat_list[-1], axis=2, keepdims=True)

                if fparams[2]:
                    feat_list[-1] /= (np.std(feat_list[-1], axis=2, keepdims=True) + 1e-6)

        if not feat_list:
            feat_list.append(x_train[:, :1, :] - x_train[:, 1:2, :])

        inputs = np.concatenate(feat_list, axis=1)

        return np.array(inputs)
