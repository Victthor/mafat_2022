
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
from sklearn.metrics import roc_auc_score
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy.signal import medfilt


@hydra.main(config_path=os.path.join('./config'), config_name='config')
def optimization_target(cfg: DictConfig):
    # print(cfg)

    pipeline = instantiate(cfg.pipeline, _recursive_=False)
    dataset = instantiate(cfg.dataset, _recursive_=False)

    min_length = 360
    offset = 360
    splitted_data = [df for df in dataset if df.shape[0] >= min_length]

    intervals = []

    # split to intervals with min_length
    for df in splitted_data:

        cur_start = 0
        cur_end = cur_start + min_length
        while cur_end < df.shape[0]:
            intervals.append(df.iloc[cur_start:cur_end])
            cur_start += offset
            cur_end = cur_start + min_length

    X_train = [interval[['RSSI_Left', 'RSSI_Right']].to_numpy().transpose() for interval in intervals]
    X_train = [vec - np.median(vec, axis=1, keepdims=True) for vec in X_train]
    # X_train = np.array(X_train)
    y_train = np.array([int(interval[['Num_People']].iloc[0]) for interval in intervals])
    y_train[y_train > 0] = 1

    # y_train[y_train == 0] = -1
    # devices = np.array([int(interval[['Device_ID']].iloc[0]) for interval in intervals])
    rooms = np.array([int(interval[['Room_Num']].iloc[0]) for interval in intervals])

    max_mins = np.array([np.max(data, axis=1) - np.min(data, axis=1) for data in X_train])
    diffs = [(data[0, :] - data[1, :]).astype(np.float32) for data in X_train]
    max_mins_ = np.array([np.max(data) - np.min(data) for data in diffs])

    text_label = [f'room_{room}_label_{label}' for room, label in zip(rooms, y_train)]
    diffs_std = [np.std(diff) for diff in diffs]

    diffs_values, diffs_counts = [], []
    for diff in diffs:
        values, counts = np.unique(diff, return_counts=True)
        diffs_values.append(values)
        diffs_counts.append(counts)
    unique_diffs_std = [np.std(diff) for diff in diffs_values]
    len_counts = [diffs_count.size for diffs_count in diffs_counts]

    self_diffs_left = [(data[0, 1:] - data[0, :-1]).astype(np.float32) for data in X_train]
    self_diffs_right = [(data[1, 1:] - data[1, :-1]).astype(np.float32) for data in X_train]
    diffs_diffs = [(data[1:] - data[:-1]).astype(np.float32) for data in diffs]

    feat_1 = [np.sum(np.abs(data)) for data in diffs_diffs]
    feat_1_l = [np.sum(np.abs(data)) for data in self_diffs_left]
    feat_1_r = [np.sum(np.abs(data)) for data in self_diffs_right]

    feat_2 = [data[data != 0].size for data in diffs_diffs]
    feat_2_l = [data[data != 0].size for data in self_diffs_left]
    feat_2_r = [data[data != 0].size for data in self_diffs_right]

    df = {
        'feat_1': feat_1,
        'feat_2': feat_2,
        'feat_1_l': feat_1_l,
        'feat_1_r': feat_1_r,
        'feat_2_l': feat_2_l,
        'feat_2_r': feat_2_r,
        'diffs_std': diffs_std,
        'unique_diffs_std': unique_diffs_std,
        'len_counts': len_counts,
        'text_label': text_label,
    }

    df = pd.DataFrame(df)
    fig = px.scatter_matrix(
        df,
        dimensions=[
            "feat_1",
            "feat_2",
            "feat_1_l",
            "feat_1_r",
            "feat_2_l",
            "feat_2_r",
            "diffs_std",
            "unique_diffs_std",
            "len_counts",
        ],
        color="text_label")
    fig.show()

    sel_class = 1
    sel_room = 4

    for data, diff, max_min, max_min_, cur_class, cur_room in zip(X_train, diffs, max_mins, max_mins_, y_train, rooms):
        if cur_class != sel_class or cur_room != sel_room:
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data[0, :], mode='lines+markers', name='RSSI_Left'))
        fig.add_trace(go.Scatter(y=data[1, :], mode='lines+markers', name='RSSI_Right'))
        fig.add_trace(go.Scatter(y=diff, mode='lines+markers', name='RSSI_Left - RSSI_Right'))
        fig.add_trace(go.Scatter(y=medfilt(diff, kernel_size=15), mode='lines+markers', name='medfilt(RSSI_Left - RSSI_Right)'))
        fig.show()

        b = 1

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=max_mins[:, 0]))
    fig.add_trace(go.Histogram(x=max_mins[:, 1]))
    fig.add_trace(go.Histogram(x=max_mins_))

    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.show()

    rooms = np.array([int(interval[['Room_Num']].iloc[0]) for interval in intervals])

    # print('fitting')
    # pipeline.fit(X_train, y_train)
    #
    # print('auc calculation')
    # auc = roc_auc_score(y_train, pipeline.decision_function(X_train), average=None)

    # auc per room
    auc_room = {}
    auc_room_test = {}
    for room in set(rooms):

        X_train_rooms = np.array([df for df_inx, df in enumerate(X_train) if rooms[df_inx] != room])
        y_train_rooms = y_train[rooms != room]

        X_test_rooms = np.array([df for df_inx, df in enumerate(X_train) if rooms[df_inx] == room])
        y_test_rooms = y_train[rooms == room]

        print(f'fitting room: {room}')
        pipeline.fit(X_train_rooms, y_train_rooms)
        auc_room[room] = roc_auc_score(y_train_rooms, pipeline.predict_proba(X_train_rooms)[:, 1], average=None)

        print(f'testing room: {room}')
        auc_room_test[room] = roc_auc_score(y_test_rooms, pipeline.predict_proba(X_test_rooms)[:, 1], average=None)

        # # auc per device
        # auc_device = {}
        # for device in set(devices):
        #     X_train_devices = [df for df_inx, df in enumerate(X_train) if devices[df_inx] == device]
        #     auc_device[device] = roc_auc_score(y_train[devices == device], pipeline.decision_function(X_train_devices), average=None)

        # print(f'auc: {auc}')

    auc_room_list = list(auc_room.values())
    auc_room_test_list = list(auc_room_test.values())
    # auc_device_list = list(auc_device.values())

    print(f'auc per room: {auc_room_list}')
    print(f'auc per room test: {auc_room_test_list}')
    # print(f'auc per device: {auc_device_list}')
    print(f'mean auc per room: {np.mean(auc_room_list)} std: {np.std(auc_room_list)}')
    print(f'mean auc per room test: {np.mean(auc_room_test_list)} std: {np.std(auc_room_test_list)}')
    # print(f'mean auc per device: {np.mean(auc_device_list)} std: {np.std(auc_device_list)}')

    return np.mean(auc_room_test_list), np.std(auc_room_test_list)


if __name__ == '__main__':

    optimization_target()
