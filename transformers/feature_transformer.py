
import numpy as np
# from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):

    # def __init__(self, min_length=360):
    #     self.min_length = min_length

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x_train = [interval[['RSSI_Left', 'RSSI_Right']].to_numpy().transpose() for interval in x]
        # x_train = [vec - np.median(vec, axis=1, keepdims=True) for vec in x_train]

        diffs = [(data[0, :] - data[1, :]).astype(np.float32) for data in x_train]

        diffs_std = [np.std(diff) for diff in diffs]

        diffs_values, diffs_counts = [], []
        for diff in diffs:
            values, counts = np.unique(diff, return_counts=True)
            diffs_values.append(values)
            diffs_counts.append(counts)
        unique_diffs_std = [np.std(diff) for diff in diffs_values]
        len_counts = [diffs_count.size for diffs_count in diffs_counts]

        self_diffs_left = [(data[0, 1:] - data[0, :-1]).astype(np.float32) for data in x_train]
        self_diffs_right = [(data[1, 1:] - data[1, :-1]).astype(np.float32) for data in x_train]
        diffs_diffs = [(data[1:] - data[:-1]).astype(np.float32) for data in diffs]

        feat_1 = [np.sum(np.abs(data)) for data in diffs_diffs]
        feat_1_l = [np.sum(np.abs(data)) for data in self_diffs_left]
        feat_1_r = [np.sum(np.abs(data)) for data in self_diffs_right]

        feat_2 = [data[data != 0].size for data in diffs_diffs]
        feat_2_l = [data[data != 0].size for data in self_diffs_left]
        feat_2_r = [data[data != 0].size for data in self_diffs_right]

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
