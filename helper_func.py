import numpy as np
# from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureTransformer(BaseEstimator, TransformerMixin):

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
            x_train = np.array([x[['RSSI_Left', 'RSSI_Right']].to_numpy().astype(np.float32).transpose(), ])

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


class TSTransformer(BaseEstimator, TransformerMixin):

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
            x_train = np.array([x[['RSSI_Left', 'RSSI_Right']].to_numpy().astype(np.float32).transpose(), ])

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


class CatchTransformer(BaseEstimator, TransformerMixin):

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
