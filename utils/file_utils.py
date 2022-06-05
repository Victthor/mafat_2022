
import os
import pandas as pd
import fnmatch
import shutil


def read_data(input_path):
    """
    Read train data, remove nan data.
    return: DataFrame.
    """
    data = pd.read_csv(input_path)
    data.dropna(subset=["Num_People"], inplace=True)
    return data


def split_and_save_data(data_path, output_path):
    # data_path = r'/home/victore/Downloads/mafat_wifi_challenge_training_set_v1.csv'
    # output_path = r'/home/victore/PycharmProjects/mafat_2022_data'
    data = read_data(data_path)

    # 1) split by device id
    for dev_id in data['Device_ID'].unique():
        data_per_dev_id = data.loc[data['Device_ID'] == dev_id]
        # 2) split by continuous same num people
        start_inx = 0
        cur_num_people = data_per_dev_id['Num_People'].iloc[start_inx]

        for row_inx in range(1, data_per_dev_id.shape[0]):
            if data_per_dev_id['Num_People'].iloc[row_inx] != cur_num_people:
                filename = f'dev_id_{dev_id}_start_inx_{start_inx}.pickle'
                print(f'saving {filename}')
                # splitted_data.append(data_per_dev_id.iloc[start_inx:row_inx])
                splitted = data_per_dev_id.iloc[start_inx:row_inx]
                splitted.to_pickle(os.path.join(output_path, filename))
                cur_num_people = data_per_dev_id['Num_People'].iloc[row_inx]
                start_inx = row_inx


def lsdirs(root_dir, wldc=None, is_full_path=False):

    # get all files and folders matching wildcard in the current directory
    all_names = os.listdir(root_dir)
    if wldc:
        all_names = fnmatch.filter(all_names, wldc)

    subdirs = []
    for filename in all_names:
        if os.path.isdir(os.path.join(root_dir, filename)):
            if is_full_path:
                filename = os.path.join(root_dir, filename)

            subdirs.append(filename)

    subdirs.sort()
    return subdirs


def newdir(dir_name, is_remove_old=False):

    if not os.path.isdir(dir_name):
        # recursively create all intermediate-level directories needed to contain the leaf directory
        os.makedirs(dir_name, exist_ok=True)
    else:
        if is_remove_old:
            shutil.rmtree(dir_name, ignore_errors=True)
            os.makedirs(dir_name, exist_ok=True)


def lsfiles(root_dir, wldc=None, is_full_path=False):
    # get all files and folders matching wldc in the current directory
    all_names = os.listdir(root_dir)
    if wldc:
        all_names = fnmatch.filter(all_names, wldc)

    files = []
    for filename in all_names:
        if os.path.isfile(os.path.join(root_dir, filename)):
            if is_full_path:
                filename = os.path.join(root_dir, filename)

            files.append(filename)

    files.sort()
    return files
