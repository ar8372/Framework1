from sklearn import metrics as skmetrics
import tensorflow as tf 
import numpy as np

class RegressionMetrics:
    def __init__(self):
        self.metrics = {
            "mae": self._mae,
            "mse": self._mse, 
            "rmse": self._rmse,
            "msle": self._msle,
            "rmsle": self._rmsle,
            "r2": self._r2,
        }
    
    def __call__(self, metric, y_true, y_pred):
        if metric not in self.metrics:
            raise Exception(f"{metrics}: Metric not implemented")
        if metric == "mae":
            return self._mae(y_true= y_true, y_pred=y_pred)
        if metric == "mse":
            return self._mse(y_true= y_true, y_pred=y_pred)
        if metric == "rmse":
            return self._rmse(y_true= y_true, y_pred=y_pred)
        if metric == "msle":
            return self._msle(y_true= y_true, y_pred=y_pred)
        if metric == "rmsle":
            return self._rmsle(y_true= y_true, y_pred=y_pred)
        if metric == "r2":
            return self._r2(y_true= y_true, y_pred=y_pred)
    
    @staticmethod
    def _mae( y_true, y_pred):
        return skmetrics.mean_absolute_error(y_true=y_true, y_pred = y_pred)

    @staticmethod
    def _mse( y_true, y_pred):
        return skmetrics.mean_squared_error(y_true=y_true, y_pred = y_pred)


    def _rmse(self, y_true, y_pred):
        return np.sqrt(self._mse(y_true, y_pred))

    @staticmethod
    def _msle( y_true, y_pred):
        return skmetrics.mean_squared_log_error(y_true=y_true, y_pred = y_pred)

    def _rmsle( self,y_true, y_pred):
        return np.sqrt(self._msle(y_true, y_pred))

    @staticmethod
    def _r2( y_true, y_pred):
        return skmetrics.r2_score(y_true=y_true, y_pred = y_pred)

class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy": self._accuracy,
            "f1": self._f1,
            "recall": self._recall,
            "precisison": self._precision,
            "auc": self._auc,
            "logloss": self._logloss,
            "auc_tf": self._auc_tf,
        }
    
    # it allows to use an instance of this class as a function 
    # a= Class..ics() then a("auc",y_true,y_pred)
    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception(f"{metric}: Metric not implemented")
        if metric == "auc":
            if y_proba is None:
                raise Exception(f"y_proba can't be None for {metric}")
            return self._auc(y_true=y_true, y_pred = y_proba)
        if metric == "logloss":
            if y_proba is None:
                raise Exception(f"y_proba can't be None for {metric}")
            return self._auc(y_true=y_true, y_pred = y_proba)
        if metric == "auc_tf":
            if y_proba is None:
                raise Exception(f"y_proba can't be None for {metric}")
            return self._auc_tf(y_true= y_true, y_pred= y_proba)
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _accuracy( y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred = y_pred)

    @staticmethod 
    def _f1( y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred = y_pred)

    @staticmethod 
    def _recall( y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred = y_pred)

    @staticmethod 
    def _precision( y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred = y_pred)

    @staticmethod 
    def _auc( y_true, y_pred):
        # auc expects probability so we need y_proba
        return skmetrics.roc_auc_score(y_true=y_true, y_score= y_pred)

    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetrics.log_loss(y_true= y_true, y_pred= y_pred)

    
    @staticmethod
    def _auc_tf(y_true, y_pred):
        # should have cuda enabled
        def fallback_auc(y_true, y_pred):
            try:
                return metrics.roc_auc_score(y_true, y_pred)
            except:
                return 0.5 
        return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)