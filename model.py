
import os
os.environ["NUMBA_CACHE_DIR"] = r"/tmp/numba_cache"

import pickle
# import numpy as np
# from os.path import isfile
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.transformations.panel.catch22 import Catch22
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV

from helper_func import FeatureTransformer, TSTransformer, CatchTransformer
# import os


class model:
    def __init__(self):
        """
        Init the model
        """

        ts_transformer_params = {
            'with_mean': None,
            'with_std': False,
            'feature_set': {
                'diff': [True, 'median', True],
                'avg': [True, 'median', True],
                'rssi_left': [True, 'median', False],
                'rssi_right': [True, 'median', True],
            },
        }

        sgd_classifier_params = {
            'loss': 'log',
            'penalty': 'l2',
            'alpha': 0.222,
            'max_iter': 500,
            'verbose': 1,
            'learning_rate': 'optimal',
            'n_iter_no_change': 150,
            'class_weight': 'balanced',
            'random_state': 1,
        }

        feat_transformer_params = {
            'with_mean': None,
            'with_std': True,
        }

        gradient_boosting_clf_params = {
            'loss': 'binary_crossentropy',
            'learning_rate': 0.06157763403218902,
            'max_iter': 100,
            'max_leaf_nodes': 31,
            'max_depth': None,
            'min_samples_leaf': 20,
            'l2_regularization': 0.0014145416042771236,
            'max_bins': 255,
            'categorical_features': None,
            'n_iter_no_change': 20,
            'random_state': 42,
            'verbose': 1,
        }

        catch_transformer_params = {
            'with_mean': None,
            # 'with_std': False,
            'sel_feat': 'diff',
        }

        catch_gradient_boosting_clf_params = {
            'loss': 'binary_crossentropy',
            'learning_rate': 0.028,
            'max_iter': 100,
            'max_leaf_nodes': 31,
            'max_depth': None,
            'min_samples_leaf': 20,
            'l2_regularization': 0.0042,
            'max_bins': 255,
            'categorical_features': None,
            'n_iter_no_change': 20,
            'random_state': 42,
            'verbose': 1,
        }

        pipeline_1 = Pipeline(
            [
                ('ts_transformer', TSTransformer(ts_transformer_params)),
                ('minirocket', MiniRocketMultivariate()),
                ('standard_scaler', StandardScaler()),
                ('sgd_classifier', SGDClassifier(**sgd_classifier_params)),
            ]
        )

        pipeline_2 = Pipeline(
            [
                ('feat_transformer', FeatureTransformer(feat_transformer_params)),
                ('gradient_boosting_clf', HistGradientBoostingClassifier(**gradient_boosting_clf_params))
            ]
        )

        pipeline_3 = Pipeline(
            [
                ('catch_trans', CatchTransformer(catch_transformer_params)),
                ('catch', Catch22(outlier_norm=False, replace_nans=True)),
                ('gradient_boosting_clf', HistGradientBoostingClassifier(**catch_gradient_boosting_clf_params))
            ]
        )

        final_estimator = Pipeline(
            [
                ('standard_scaler', StandardScaler()),
                ('logistic_regression', LogisticRegressionCV(class_weight='balanced'))
            ]
        )

        self.model = StackingClassifier(
            estimators=[('pipe1', pipeline_1), ('pipe2', pipeline_2), ('pipe3', pipeline_3)],
            final_estimator=final_estimator,
            cv=5,
            stack_method='predict_proba',
            verbose=1,
        )

    def predict(self, X):
        """
        Edit this function to fit your model.

        This function should provide predictions of labels on (test) data.
        Make sure that the predicted values are in the correct format for the scoring
        metric.
        preprocess: if our code for add feature to the data before we predict the model.
        :param X: is DataFrame with the columns - 'Time', 'Device_ID', 'Rssi_Left','Rssi_Right'.
                  X is window of size 360 samples time, shape(360,4).
        :return: a float value of the prediction for class 1 (the room is occupied).
        """
        # preprocessing should work on a single window, i.e a dataframe with 360 rows and 4 columns
        # X = preprocess(X, self.RSSI_value_selection)
        y = self.model.predict_proba(X)[:, 1][0]

        """
        Track 2 - for track 2 we naively assume that the model from track-1 predicts 0/1 correctly. 
        We use that assumption in the following way:
        when the room is occupied (1,2,3 - model predicted 1) we assign the majority class (2) as prediction.       
        """
        # y = 0 if y<0.5 else 2
        return y

    def load(self, dir_path):
        """
        Edit this function to fit your model.

        This function should load the model that you trained on the train set.
        :param dir_path: A path for the folder the model is submitted
        """
        model_name = 'model_track_1.sav'
        model_file = os.path.join(dir_path, model_name)
        self.model = joblib.load(model_file)

    def save(self, dir_path):
        filename = "model_track_1.sav"
        with open(os.path.join(dir_path, filename), 'wb') as f:
            pickle.dump(self.model, f)
