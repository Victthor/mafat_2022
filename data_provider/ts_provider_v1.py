
import numpy as np
import random


class Provider:
    def __init__(self, params, dataset=None):
        self.min_length = params.get('min_length', 360)
        self.offset = params.get('offset', 360)
        self.shuffle = params.get('shuffle', False)
        self.seed = params.get('seed', 42)

        self.intervals = self.fill_intervals(dataset)

    def fill_intervals(self, dataset):
        intervals = []
        splitted_data = [df for df in dataset if df.shape[0] >= self.min_length]

        # split to intervals with min_length
        for df in splitted_data:

            cur_start = 0
            cur_end = cur_start + self.min_length
            while cur_end < df.shape[0]:
                intervals.append(df.iloc[cur_start:cur_end])
                cur_start += self.offset
                cur_end = cur_start + self.min_length

        if self.shuffle:
            random.shuffle(intervals)

        return intervals

    def prepare_io(self):

        outputs = np.array([int(interval[['Num_People']].iloc[0]) for interval in self.intervals])
        outputs[outputs > 0] = 1

        return self.intervals, outputs

    def get_by_col_name(self, col_name='Room_Num'):
        return np.array([int(interval[[col_name]].iloc[0]) for interval in self.intervals])
