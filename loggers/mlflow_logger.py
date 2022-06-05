
from mlflow_extend import mlflow
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from datetime import datetime

from utils.misc_utils import isfloat, isint


class MLFlowLogger:

    def __init__(self, params):

        self.start_run = mlflow.start_run
        self.log_figure = mlflow.log_figure
        self.log_dict = mlflow.log_dict
        self.log_param = mlflow.log_param
        self.log_metric = mlflow.log_metric

        self.tracking_uri = params.get('tracking_uri', '')

        try:
            self.hydra_config = HydraConfig.get()
        except ValueError:
            print('HydraConfig was not set, setting to None')
            self.hydra_config = None

        mlflow.set_tracking_uri(self.tracking_uri)

    def set_experiment(self, experiment_name):
        mlflow.set_experiment(experiment_name=experiment_name)

    @staticmethod
    def log_config(cfg, name):
        cfg_dict = OmegaConf.to_container(cfg)
        mlflow.log_dict(cfg_dict, name)

    def log_overrides(self):
        if self.hydra_config is not None:
            overrides = self.hydra_config.overrides['task']

            for param in overrides:
                name, val = param.split('=')
                if isfloat(val):
                    self.log_param(name, float(val))
                elif isint(val):
                    self.log_param(name, int(val))
                else:
                    self.log_param(name, val)

    def get_run_name(self):
        if self.hydra_config is not None and self.hydra_config.overrides['task']:
            run_name = self.hydra_config.sweep.dir.replace('/', '_') + \
                '_cur_run_' + datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        else:
            if self.hydra_config is not None:
                run_name = self.hydra_config.run.dir.replace('/', '_')
            else:
                run_name = 'set run name!'

        return run_name
