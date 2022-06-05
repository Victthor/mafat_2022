
import hydra
from omegaconf import ListConfig
from sklearn.pipeline import Pipeline


def pipeline_factory(steps_config: ListConfig) -> Pipeline:

    steps = []

    for step_config in steps_config:

        step_name, step_params = list(step_config.items())[0]

        pipeline_step = (step_name, hydra.utils.instantiate(step_params))
        steps.append(pipeline_step)

    return Pipeline(steps)
