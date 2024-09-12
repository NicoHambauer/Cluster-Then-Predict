import warnings

import sklearn.cluster as c
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import os
from exceptions.custom_exceptions import BadClusteringError

#ignore ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class ClusterThenPredict:
    # amount logistic regressions as prediction models
    # choose clustering algorithm from scikitlearn clusters

    def _get_clustering(self, clustering, arg_dict):
        if clustering == "kmeans":
            self.clustering = c.KMeans(n_clusters=self.n_clusters, max_iter=arg_dict['max_iter'], n_init=arg_dict['n_init'], init=arg_dict['init'], algorithm=arg_dict['algorithm'], tol=arg_dict['tol'], random_state=self.random_state)
        elif clustering == "spectral":
            self.clustering = c.SpectralClustering(n_clusters=self.n_clusters, random_state=self.random_state, eigen_solver='amg', affinity='rbf')
        elif clustering == "agglomerative":
            self.clustering = c.AgglomerativeClustering(n_clusters=self.n_clusters, linkage=arg_dict['linkage'])
        elif clustering == "DBSCAN":
            self.n_clusters_not_used = True
            # print("Warning: n_clusters parameter is not used for DBSCAN.")
            if self.verbose == 2:
                print("eps:", arg_dict['eps'], "min_samples", arg_dict['min_samples'], "metric", arg_dict['metric'], "algorithm", arg_dict['algorithm'])
            self.clustering = c.DBSCAN(eps=arg_dict['eps'], min_samples=arg_dict['min_samples'], metric=arg_dict['metric'], algorithm=arg_dict['algorithm'])
        elif clustering == "OPTICS":
            self.n_clusters_not_used = True
            if self.verbose == 2:
                print("Warning: n_clusters parameter is not used for OPTICS.")
            if arg_dict['max_eps'] == 'inf':
                arg_dict['max_eps'] = float('inf')
            self.clustering = c.OPTICS(max_eps=arg_dict['max_eps'], xi=arg_dict['xi'], min_samples=arg_dict['min_samples'], n_jobs=-1)
        elif clustering == "Birch":
            self.clustering = c.Birch(n_clusters=self.n_clusters, threshold=arg_dict['threshold'], branching_factor=arg_dict['branching_factor'])
        elif clustering == "MeanShift":
            self.n_clusters_not_used = True
            if self.verbose == 2:
                print("Warning: n_clusters parameter is not used for MeanShift.")
            self.clustering = c.MeanShift()
        elif clustering == "AffinityPropagation":
            self.n_clusters_not_used = True
            if self.verbose == 2:
                print("Warning: n_clusters parameter is not used for AffinityPropagation.")
            self.clustering = c.AffinityPropagation(random_state=self.random_state)
        else:
            raise ValueError("Unknown clustering algorithm: {}".format(clustering))

    def __init__(self, n_clusters=3, random_state=42, clustering="kmeans", task="classification", arg_dict=None, fold=0, dataset_name="", verbose=2):
        self.verbose = verbose
        self.n_clusters = n_clusters if n_clusters is not None else 1
        self.random_state=random_state
        self.predictors = []
        self.task=task
        self.n_clusters_not_used = False
        self.lr_alpha = arg_dict['lr_alpha'] if 'lr_alpha' in arg_dict.keys() else None
        self.C = arg_dict['C'] if 'C' in arg_dict.keys() else None
        self._get_clustering(clustering=clustering, arg_dict=arg_dict)
        self.clustering_name = clustering


        self.fold = fold
        self.dataset_name = dataset_name

        for i in range(self.n_clusters):
            if self.task == "classification":
                self.predictors.append(LogisticRegression(random_state=self.random_state+i, C=self.C, solver=arg_dict["solver"], penalty=arg_dict["penalty"], max_iter=300))
            elif self.task == "regression":
                self.predictors.append(Ridge(random_state=self.random_state+i, alpha=self.lr_alpha))
            else:
                raise ValueError("Unknown task: {}".format(self.task))
        self.only_one_class = dict()


    def train_supervised_model(self, X, labels):
        # needed because not all clustering algorithms support prediction after fit
        self.supervised_model = RandomForestClassifier(random_state=self.random_state)
        self.supervised_model.fit(X, labels)
        self.predict_label = self.supervised_model.predict
        

    def _get_cluster_labels(self,X):
        if isinstance(X, pd.DataFrame):
            X_with_labels = X.copy()
            X_with_labels['cluster_labels'] = self.clustering.labels_
            labels = X_with_labels['cluster_labels']
        else:
            X_with_labels = np.column_stack([X, self.clustering.labels_])
            labels = X_with_labels[:, -1]
        if hasattr(self.clustering, 'predict'):
            self.predict_label = self.clustering.predict
        else:
            # Train a RandomForestClassifier if 'predict' is not available
            # if isinstance(X, pd.DataFrame):
            #     cluster_data = X_with_labels.copy()
            #     X_no_labels = cluster_data.drop('cluster_labels', axis=1)
            #     self.train_supervised_model(X, self.clustering.labels_)
            # else:
            self.train_supervised_model(X, self.clustering.labels_) # X_with_labels[:, :-1], self.clustering.labels_)
        return X_with_labels, labels


    def _initialize_predictors_no_n_clusters(self):
        self.predictors = []
        labels_set = set(self.clustering.labels_) - {-1}
        for i in range(len(labels_set)):
            if self.task == 'regression':
                self.predictors.append(Lasso(random_state=self.random_state+i, alpha=self.lr_alpha))
            else:
                self.predictors.append(LogisticRegression(random_state=self.random_state+i))
        if -1 in self.clustering.labels_:
            if self.task == 'regression':
                self.predictors.append(Lasso(random_state=self.random_state, alpha=self.lr_alpha))
            else:
                self.predictors.append(LogisticRegression(random_state=self.random_state))


    # find clusters and train log regressions
    def fit(self, X, y, force_retrain=False):
        self.clustering.fit(X)

        X_with_labels, labels = self._get_cluster_labels(X=X)

        unique_clusters = len(np.unique(labels))

        if self.verbose == 2:
            print(f"Unique clusters: {unique_clusters}")

        if (unique_clusters <= 1 or unique_clusters > 50) and not force_retrain:
            raise BadClusteringError(f"Inappropriate cluster count: {unique_clusters}. Adjust parameters.")

        if self.n_clusters_not_used:
            self._initialize_predictors_no_n_clusters()

        if -1 in np.unique(labels):
            indices = np.where(labels == -1)[0]
            print(f"Number of outliers: {len(indices)}")
            if isinstance(X, pd.DataFrame):
                cluster_data = X_with_labels.iloc[indices]
                X_cluster = cluster_data.drop('cluster_labels', axis=1)
                y_cluster = y.iloc[indices]
            else:
                cluster_data = X_with_labels[labels == -1]
                X_cluster = cluster_data[:, :-1]
                y_cluster = y[labels == -1]
            if len(np.unique(y_cluster)) < 2:
                self.only_one_class[-1] = np.unique(y_cluster)[0]
            else: # can't continue here as it is no loop like below
                try:
                    self.predictors[-1].fit(X_cluster, y_cluster)
                except ConvergenceWarning as e:
                    if self.verbose == 2:
                        print(e)
                        warnings.warn("ConvergenceWarning: The optimizer did not converge. Increase the number of iterations.")
                current_path = f"test_for_HICSS_00\\{self.task}\\{self.dataset_name}\\{self.clustering_name}\\{self.fold}"
                plt.scatter(y = y_cluster, x = np.linspace(np.min(X_cluster), np.max(X_cluster), len(X_cluster)))
                if not os.path.exists(current_path):
                    os.makedirs(current_path)
                plt.savefig(f"{current_path}\\{-1}.png")
                plt.close()

            enum = enumerate(self.predictors[:-1])
        else:
            enum = enumerate(self.predictors)
        for i, predictor in enum:
            indices = np.where(labels == i)[0]
            if isinstance(X, pd.DataFrame):
                cluster_data = X_with_labels.iloc[indices]
                X_cluster = cluster_data.drop('cluster_labels', axis=1)
                y_cluster = y.iloc[indices]
            else:
                cluster_data = X_with_labels[labels == i]
                X_cluster = cluster_data[:, :-1]
                y_cluster = y[labels == i]
            if len(np.unique(y_cluster)) < 2:
                self.only_one_class[i] = np.unique(y_cluster)[0]
                continue

            try:
                predictor.fit(X_cluster, y_cluster)
            except ConvergenceWarning as e:
                if self.verbose == 2:
                    print(e)
                    warnings.warn("ConvergenceWarning: The optimizer did not converge. Increase the number of iterations.")
            # To find outliers:
            plt.scatter(y = y_cluster, x = np.linspace(-250, 250, len(y_cluster)))
            plt.title(f"Number of samples = {len(indices)}")
            current_path = f"test_for_HICSS_00\\{self.task}\\{self.dataset_name}\\{self.clustering_name}\\{self.fold}"
            if not os.path.exists(current_path):
                os.makedirs(current_path)
            plt.savefig(f"{current_path}\\{i}.png")
            plt.close()

    def predict(self, X):
        cluster_labels = self.predict_label(X)
        predictions = np.empty(len(X), dtype=float)

        if -1 in np.unique(cluster_labels): 
            # if condition is not met, it can enumerate of all as it chengs for len(cluster_indices) down below, even if there were outliers during training
            enum = enumerate(self.predictors[:-1])

            cluster_indices = np.where(cluster_labels == -1)[0]
            if -1 in self.only_one_class.keys():
                predictions[cluster_indices] = self.only_one_class[-1]
            elif len(cluster_indices) > 0:
                if isinstance(X, pd.DataFrame):
                    X_cluster = X.iloc[cluster_indices]
                else:
                    X_cluster = X[cluster_indices, :]
                cluster_predictions = self.predictors[-1].predict(X_cluster)
                predictions[cluster_indices] = cluster_predictions.flatten()
        else:
            enum = enumerate(self.predictors)

        for i, predictor in enum:
            cluster_indices = np.where(cluster_labels == i)[0]
            if i in self.only_one_class.keys():
                predictions[cluster_indices] = self.only_one_class[i]
            elif len(cluster_indices) > 0:
                if isinstance(X, pd.DataFrame):
                    X_cluster = X.iloc[cluster_indices]
                else:
                    X_cluster = X[cluster_indices, :]
                cluster_predictions = predictor.predict(X_cluster)
                predictions[cluster_indices] = cluster_predictions.flatten()
        return predictions


    def predict_proba(self, X):
        cluster_labels = self.predict_label(X)
        predictions = np.empty(len(X), dtype=float)

        if -1 in np.unique(cluster_labels):

            # if condition is not met, it can enumerate of all as it chengs for len(cluster_indices) down below, even if there were outliers during training
            enum = enumerate(self.predictors[:-1])

            cluster_indices = np.where(cluster_labels == -1)[0]
            if -1 in self.only_one_class.keys():
                predictions[cluster_indices] = self.only_one_class[-1]
            elif len(cluster_indices) > 0:
                if isinstance(X, pd.DataFrame):
                    X_cluster = X.iloc[cluster_indices]
                else:
                    X_cluster = X[cluster_indices, :]
                cluster_predictions = self.predictors[-1].predict_proba(X_cluster)[:, 1] # predict proba with class 1 likelihood
                predictions[cluster_indices] = cluster_predictions.flatten()
        else:
            enum = enumerate(self.predictors)

        for i, predictor in enum:
            cluster_indices = np.where(cluster_labels == i)[0]
            if i in self.only_one_class.keys():
                predictions[cluster_indices] = self.only_one_class[i]
            elif len(cluster_indices) > 0:
                if isinstance(X, pd.DataFrame):
                    X_cluster = X.iloc[cluster_indices]
                else:
                    X_cluster = X[cluster_indices, :]
                cluster_predictions = predictor.predict_proba(X_cluster)[:, 1] # predict proba with class 1 likelihood
                predictions[cluster_indices] = cluster_predictions.flatten()
        return predictions
    
    