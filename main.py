
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
from sklearn.metrics import roc_auc_score


@hydra.main(config_path=os.path.join('./config'), config_name='config')
def optimization_target(cfg: DictConfig):
    # print(cfg)

    pipeline = instantiate(cfg.pipeline, _recursive_=False)
    dataset = instantiate(cfg.dataset, _recursive_=False)
    logger = instantiate(cfg.logger)
    provider = instantiate(cfg.data_provider, dataset=dataset)

    logger.set_experiment(cfg.experiment_name)
    run_name = logger.get_run_name()

    rooms = provider.get_by_col_name(col_name='Room_Num')
    inputs, outputs = provider.prepare_io()

    with logger.start_run(run_name=run_name):

        logger.log_config(cfg, 'config.yaml')
        logger.log_overrides()

        # auc per room
        auc_room = {}
        auc_room_test = {}
        for room in set(rooms):

            X_train_rooms = [df for df_inx, df in enumerate(inputs) if rooms[df_inx] != room]
            y_train_rooms = outputs[rooms != room]

            X_test_rooms = [df for df_inx, df in enumerate(inputs) if rooms[df_inx] == room]
            y_test_rooms = outputs[rooms == room]

            print(f'fitting room: {room}')
            pipeline.fit(X_train_rooms, y_train_rooms)
            auc_room[room] = roc_auc_score(y_train_rooms, pipeline.predict_proba(X_train_rooms)[:, 1], average=None)
            logger.log_metric(f'train room {room} auc', auc_room[room])

            print(f'testing room: {room}')
            auc_room_test[room] = roc_auc_score(y_test_rooms, pipeline.predict_proba(X_test_rooms)[:, 1], average=None)
            logger.log_metric(f'test room {room} auc', auc_room_test[room])

        auc_room_list = list(auc_room.values())
        auc_room_test_list = list(auc_room_test.values())

        print(f'auc per room: {auc_room_list}')
        print(f'auc per room test: {auc_room_test_list}')
        print(f'mean auc per room: {np.mean(auc_room_list)} std: {np.std(auc_room_list)}')
        print(f'mean auc per room test: {np.mean(auc_room_test_list)} std: {np.std(auc_room_test_list)}')

        logger.log_metric(f'mean train room auc', np.mean(auc_room_list))
        logger.log_metric(f'std train room auc', np.std(auc_room_list))
        logger.log_metric(f'mean test room auc', np.mean(auc_room_test_list))
        logger.log_metric(f'std test room auc', np.std(auc_room_test_list))

    return np.mean(auc_room_test_list)


if __name__ == '__main__':

    optimization_target()
