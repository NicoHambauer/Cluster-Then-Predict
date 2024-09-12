import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
# from piml.models import GAMINetRegressor, GAMINetClassifier
# from interpret.glassbox import (
#     ExplainableBoostingClassifier,
#     ExplainableBoostingRegressor,
# )
# from pygam import terms, s, f
# from pygam.pygam import LogisticGAM, LinearGAM
# from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

# from baseline.exnn.exnn import ExNN
# from baseline.lemeln_nam.nam.wrapper import NAMClassifier, NAMRegressor
# from igann import IGANN
import torch

from cluster_then_predict_models import ClusterThenPredict
from logit_leaf_models import LLM_Classifier, LLM_Regressor
class Model:

    def __init__(self, model_name, task, arg_dict, num_cols=None, cat_cols=None, fold=0, dataset_name="", verbose=2):
        self.model_name = model_name
        self.task = task
        self.arg_dict = arg_dict
        self.num_cols = num_cols
        self.cat_cols = cat_cols

        self.fold=fold
        self.dataset_name=dataset_name

        self.verbose = verbose

        self.model = self._get_model()
        # some models like exnn and GAMINET need to know which columns are categorical and which are numerical for a
        # reason internally. Other models do not need to know.

    def fit(self, X_train, y_train, force_retrain=False):
        if "DBSCAN" in self.model_name:
            cluster_algorithm = self.model_name.split('_')[1] # assuming model named like CLUSTERING_kmeans
            self.model = ClusterThenPredict(
                n_clusters=self.arg_dict["n_clusters"],
                random_state=0,
                clustering=cluster_algorithm,
                task=self.task,
                arg_dict=self.arg_dict
            )
        if isinstance(self.model, ClusterThenPredict):
            self.model.fit(X_train, y_train, force_retrain=force_retrain)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def _get_model(self,):

        if "LR" in self.model_name:
            if self.task == "classification":
                if self.arg_dict["penalty"] == "elasticnet":
                    return LogisticRegression(
                        C=self.arg_dict["C"],
                        penalty=self.arg_dict["penalty"],
                        class_weight=self.arg_dict["class_weight"],
                        solver=self.arg_dict["solver"],
                        l1_ratio=self.arg_dict["l1_ratio"],
                        max_iter=self.arg_dict["max_iter"],
                        n_jobs=-1,
                        random_state=0,
                    )
                else:
                    return LogisticRegression(
                        C=self.arg_dict["C"],
                        penalty=self.arg_dict["penalty"],
                        class_weight=self.arg_dict["class_weight"],
                        solver=self.arg_dict["solver"],
                        max_iter=self.arg_dict["max_iter"],
                        n_jobs=-1,
                        random_state=0,
                    )
        elif "ELASTICNET" in self.model_name:
            if self.task == "regression":
                # ridge regression would be the default, so we need to set l1_ratio to 0 in case of default
                # we utilize elasticnet with l1 ratio of [0, 1] to fit lasso, ridge and everything between
                return ElasticNet(
                    alpha=self.arg_dict["alpha"],
                    l1_ratio=self.arg_dict["l1_ratio"],
                    max_iter=2000,
                    random_state=0,
                )
        elif "RF" in self.model_name:
            if self.task == "classification":
                return RandomForestClassifier(
                    n_estimators=self.arg_dict["n_estimators"],
                    max_depth=self.arg_dict["max_depth"],
                    class_weight=self.arg_dict["class_weight"],
                    n_jobs=-1,
                    random_state=0,
                )
            elif self.task == "regression":
                return RandomForestRegressor(
                    n_estimators=self.arg_dict["n_estimators"],
                    max_depth=self.arg_dict["max_depth"],
                    n_jobs=-1,
                    random_state=0,
                )

        elif "DT" in self.model_name:
            if self.task == "classification":
                return DecisionTreeClassifier(
                    max_depth=self.arg_dict["max_depth"],
                    max_leaf_nodes=self.arg_dict["max_leaf_nodes"],
                    class_weight=self.arg_dict["class_weight"],
                    splitter=self.arg_dict["splitter"],
                    random_state=0,
                )

            elif self.task == "regression":
                return DecisionTreeRegressor(
                    max_depth=self.arg_dict["max_depth"],
                    max_leaf_nodes=self.arg_dict["max_leaf_nodes"],
                    splitter=self.arg_dict["splitter"],
                    random_state=0,
                )

        elif "XGB" in self.model_name:
            if self.task == "classification":
                return XGBClassifier(
                    n_estimators=self.arg_dict["n_estimators"],
                    max_depth=self.arg_dict["max_depth"],
                    learning_rate=self.arg_dict["learning_rate"],
                    random_state=0,
                )
            elif self.task == "regression":
                return XGBRegressor(
                    n_estimators=self.arg_dict["n_estimators"],
                    max_depth=self.arg_dict["max_depth"],
                    learning_rate=self.arg_dict["learning_rate"],
                    random_state=0,
                )

        elif "CATBOOST" in self.model_name:
            if self.task == "classification":
                return CatBoostClassifier(
                    random_seed=0,
                    task_type="GPU",
                    n_estimators=self.arg_dict["n_estimators"],
                    eta=self.arg_dict["eta"],
                    max_depth=self.arg_dict["max_depth"],
                )
            elif self.task == "regression":
                return CatBoostRegressor(
                    random_seed=0,
                    task_type="GPU",
                    n_estimators=self.arg_dict["n_estimators"],
                    eta=self.arg_dict["eta"],
                    max_depth=self.arg_dict["max_depth"],
                )
        elif "CLUSTERING" in self.model_name:
            cluster_algorithm = self.model_name.split('_')[1] # assuming model named like CLUSTERING_kmeans
            return ClusterThenPredict(
                n_clusters=self.arg_dict["n_clusters"],
                random_state=0,
                clustering=cluster_algorithm,
                task=self.task,
                arg_dict=self.arg_dict,
                fold=self.fold,
                dataset_name=self.dataset_name,
                verbose=self.verbose
            )
        elif "LLM_Regressor" in self.model_name: # added before LLM, since in string would still be true for other model
            return LLM_Regressor(
                max_depth=self.arg_dict["max_depth"],
                lr_alpha=self.arg_dict['lr_alpha'],
                ccp_alpha=self.arg_dict['ccp_alpha'],
                min_leaf_size=self.arg_dict['min_leaf_size'],
                ffs=False,
                random_state = 0
                )
        elif "LLM" in self.model_name:
            return LLM_Classifier(
                max_depth=self.arg_dict["max_depth"],
                C = self.arg_dict['C'],
                ccp_alpha=self.arg_dict['ccp_alpha'],
                min_leaf_size=self.arg_dict['min_leaf_size'], 
                solver='liblinear', 
                ffs=False,
                random_state=0,
                )
        else:
            raise ValueError("Model not supported")

    def _optimize_threshold(self, y_proba): # Frage: was macht diese Funktion genau? @Nico
        fpr, tpr, trs = roc_curve(self.y_train, y_proba)

        roc_scores = []
        thresholds = []
        for thres in trs:
            thresholds.append(thres)
            y_pred = np.where(y_proba > thres, 1, 0)
            # Apply desired utility function to y_preds, for example accuracy.
            roc_scores.append(roc_auc_score(self.y_train.squeeze(), y_pred.squeeze()))
        # convert roc_scores to numpy array
        roc_scores = np.array(roc_scores)
        # get the index of the best threshold
        ix = np.argmax(roc_scores)
        # get the best threshold
        return thresholds[ix]
