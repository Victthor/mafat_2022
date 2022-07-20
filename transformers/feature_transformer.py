
import numpy as np
# from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, params):
        self.params = params
        self.with_mean = params.get('with_mean', 'mean')
        self.with_std = params.get('with_std', False)

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

        diffs = x_train[:, 0, :] - x_train[:, 1, :]

        diffs_std = np.std(diffs, axis=1)

        diffs_values, diffs_counts = [], []
        for diff in diffs:
            values, counts = np.unique(diff, return_counts=True)
            diffs_values.append(values)
            diffs_counts.append(counts)
        unique_diffs_std = [np.std(diff) for diff in diffs_values]
        len_counts = [diffs_count.size for diffs_count in diffs_counts]

        self_diffs_left = x_train[:, 0, 1:] - x_train[:, 0, :-1]
        self_diffs_right = x_train[:, 1, 1:] - x_train[:, 1, :-1]
        diffs_diffs = diffs[:, 1:] - diffs[:, :-1]

        feat_1 = np.sum(np.abs(diffs_diffs), axis=1)
        feat_1_l = np.sum(np.abs(self_diffs_left), axis=1)
        feat_1_r = np.sum(np.abs(self_diffs_right), axis=1)

        feat_2 = np.sum(np.abs(diffs_diffs != 0), axis=1)
        feat_2_l = np.sum(np.abs(self_diffs_left != 0), axis=1)
        feat_2_r = np.sum(np.abs(self_diffs_right != 0), axis=1)

        inputs = np.stack(
            [
                feat_1,
                feat_2,
                feat_1_l,
                feat_1_r,
                feat_2_l,
                feat_2_r,
                diffs_std,
                unique_diffs_std,
                len_counts
            ],
            axis=1,
        )

        return inputs
