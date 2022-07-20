
import numpy as np
# from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, params):
        self.params = params
        self.with_mean = params.get('with_mean', 'mean')
        self.with_std = params.get('with_std', False)
        self.sel_feat = params.get('sel_feat', 'diff')

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

        # if self.with_std:
        #     x_train /= (np.std(x_train, axis=2, keepdims=True) + 1e-3)

        if self.sel_feat == 'diff':
            x_train = x_train[:, :1, :] - x_train[:, 1:2, :]
        elif self.sel_feat == 'avg':
            x_train = x_train[:, :1, :] + x_train[:, 1:2, :] / 2.0
        # elif self.sel_feat == 'rssi_left':
        #     x_train = x_train[:, :1, :]
        # elif self.sel_feat == 'rssi_right':
        #     x_train = x_train[:, 1:2, :]
        # elif self.sel_feat == 'rssi_left_rssi_right':
        #     pass

        return x_train
