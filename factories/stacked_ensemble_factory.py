
from factories.pipeline_factory import pipeline_factory
from sklearn.ensemble import StackingClassifier


def ensemble_factory(pipelines, final_estimator, cv, stack_method):
    estimators = [(f'{inx}', pipeline_factory(pipeline)) for inx, pipeline in enumerate(list(pipelines))]
    final_estimator = pipeline_factory(final_estimator)

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        stack_method=stack_method,
    )

    return clf
