
from utils.file_utils import lsfiles
import pandas as pd


def dataset_factory(input_path):

    file_names = lsfiles(input_path, wldc='*.pickle', is_full_path=True)
    splitted_data = [pd.read_pickle(file_name) for file_name in file_names]

    return splitted_data
