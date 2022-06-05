
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
from sklearn.metrics import roc_auc_score
# import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path=os.path.join('./config'), config_name='config')
def optimization_target(cfg: DictConfig):
    # print(cfg)

    pipeline = instantiate(cfg.pipeline, _recursive_=False)
    dataset = instantiate(cfg.dataset, _recursive_=False)

    min_length = 360
    offset = 180
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

    X_train = [interval[['RSSI_Left', 'RSSI_Right']] for interval in intervals]
    y_train = np.array([int(interval[['Num_People']].iloc[0]) for interval in intervals])
    y_train[y_train > 0] = 1
    # devices = np.array([int(interval[['Device_ID']].iloc[0]) for interval in intervals])
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
        print(f'fitting room: {room}')

        X_train_rooms = [df for df_inx, df in enumerate(X_train) if rooms[df_inx] != room]
        y_train_rooms = y_train[rooms != room]
        pipeline.fit(X_train_rooms, y_train_rooms)
        auc_room[room] = roc_auc_score(y_train_rooms, pipeline.decision_function(X_train_rooms), average=None)

        X_test_rooms = [df for df_inx, df in enumerate(X_train) if rooms[df_inx] == room]
        y_test_rooms = y_train[rooms == room]
        auc_room_test[room] = roc_auc_score(y_test_rooms, pipeline.decision_function(X_test_rooms), average=None)

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

    # return auc


if __name__ == '__main__':

    optimization_target()
