import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import torch
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier, export_text, DecisionTreeRegressor
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from torch import nn


class LLM_Classifier:
    """
      A EJOR LLM like python version of the LogitLeafModel.
      It uses a DT to split the data into leaf nodes and then fits a LR in each leaf node.
      """

    def __init__(self, max_depth=4, C=1.0, penalty='l1', solver='liblinear', min_leaf_size=0.05, ccp_alpha=0.0, ffs=True, random_state=42):
        self.name = 'LLM'
        self.max_depth = max_depth
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.min_leaf_size = min_leaf_size
        self.ccp_alpha = ccp_alpha
        self.ffs = ffs

        self.dt = None
        self.lm_dict = {}
        self.leaf_idx = None
        self.debug = False
        self.random_state = random_state

        self.avg_n_features_selected = None
        """ Average Number of features selected by the LR model across the leaves. """
        self.median_n_features_selected = None
        """ Median Number of features selected by the LR model across the leaves. """

        self.sum_n_features_selected = None
        """ Sum of features selected by the LR model across the leaves. Equals the number of parameters."""
        self.dt_features_used = None
        """ Number of features used by the Decision Tree. """


        self.avg_samples_per_leaf = None
        """ the average number of samples per leaf node based on the train statistics during .fit(). May be useful
        to see how stable the LLM can be on your data. """
        self.median_samples_per_leaf = None
        """ the median number of samples per leaf node based on the train statistics during .fit(). May be useful
        to see how stable the LLM can be on your data. """
        self.min_samples_per_leaf = None
        """ the minimum number of samples per leaf node based on the train statistics during .fit(). May be useful
        to see how stable the LLM can be on your data. """
        self.max_samples_per_leaf = None
        """ the maximum number of samples per leaf node based on the train statistics during .fit(). May be useful
        to see how stable the LLM can be on your data. """


    def processSubset(self, X, y, feature_set):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = sm.Logit(y, sm.add_constant(X[list(feature_set)]))
                regr = model.fit(disp=0)  # disp=0 suppresses convergence messages
                AIC = regr.aic
        # ccath LinAlgError and also PerfectSeparationError
        except np.linalg.LinAlgError:
            AIC = np.inf  # assign a "bad" AIC score
            regr = None
        except PerfectSeparationError:
            AIC = np.inf
            regr = None

        return {'model': regr, 'AIC': AIC}

    def getBest(self, X, y, significance_level=0.10):
        remaining_features = list(X.columns)
        selected_features = []
        current_score = np.inf  # AIC is to be minimized
        current_log_likelihood = None
        while remaining_features:
            scores_with_candidates = [(self.processSubset(X, y, selected_features + [candidate]), candidate) for
                                      candidate in remaining_features]
            best_result, best_candidate = min(scores_with_candidates, key=lambda item: item[0]['AIC'])
            if current_log_likelihood is not None:
                test_statistic = -2 * (current_log_likelihood - best_result['model'].llf)
                p_value = chi2.sf(test_statistic, df=1)  # sf is the survival function, which is 1 - cdf
                if p_value < significance_level:
                    remaining_features.remove(best_candidate)
                    selected_features.append(best_candidate)
                    current_score = best_result['AIC']
                    current_log_likelihood = best_result['model'].llf
                else:
                    break
            else:
                remaining_features.remove(best_candidate)
                selected_features.append(best_candidate)
                current_score = best_result['AIC']
                current_log_likelihood = best_result['model'].llf
        return {'model': best_result['model'], 'AIC': current_score, 'features': selected_features}

    def fit(self, X, y):
        self.dt = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state,
                                         min_samples_leaf=self.min_leaf_size,
                                         ccp_alpha=self.ccp_alpha)
        self.dt.fit(X, y)

        if self.debug:
            r = export_text(self.dt)
            print(r)

        self.leaf_idx = self.dt.apply(X)

        # Calculate the number of samples in each leaf, for that use all indices that are leaf nodes
        # and access the n_node_samples attribute of the sklearn tree there
        leaf_counts = self.dt.tree_.n_node_samples[np.unique(self.leaf_idx)]

        # Compute the average and median number of samples in the leaves
        self.avg_samples_per_leaf = np.mean(leaf_counts)
        self.median_samples_per_leaf = np.median(leaf_counts)
        self.min_samples_per_leaf = np.min(leaf_counts)
        self.max_samples_per_leaf = np.max(leaf_counts)

        for idx in np.unique(self.leaf_idx):
            X_leaf = X[self.leaf_idx == idx]
            y_leaf = y[self.leaf_idx == idx]

            if len(np.unique(y_leaf)) == 1:
                m = DummyClassifier(np.array(y_leaf)[0])
                # Save the model and feature names to lm_dict
                self.lm_dict[idx] = {'model': m, 'features': X_leaf.columns}
            else:

                # Fit logistic regression model on selected features
                m = LogisticRegression(penalty=self.penalty, C=self.C, solver=self.solver,
                                       max_iter=1000, random_state=self.random_state)
                if self.ffs:
                    # Perform AIC-based feature selection
                    try:
                        best_model = self.getBest(X_leaf, y_leaf)
                        X_leaf = X_leaf[best_model['features']]
                        # Save the model and feature names to lm_dict
                        self.lm_dict[idx] = {'model': m, 'features': best_model['features']}
                    except AttributeError as e:
                        print(e)
                        print('No features selected for leaf node {}'.format(idx))
                        # Save the model and feature names to lm_dict
                        self.lm_dict[idx] = {'model': m, 'features': X_leaf.columns}

                else:
                    # Save the model and feature names to lm_dict
                    self.lm_dict[idx] = {'model': m, 'features': X_leaf.columns}

                self.lm_dict[idx]['model'].fit(X_leaf, y_leaf)

            self.lm_dict[idx]['max_x_values'] = np.max(X_leaf, axis=0)
            self.lm_dict[idx]['min_x_values'] = np.min(X_leaf, axis=0)

        n_features_selected = []
        feature_names_selected = []
        for idx in np.unique(self.leaf_idx):
            m = self.lm_dict[idx]
            model_name = m['model'].name if hasattr(m['model'], 'name') else m['model'].__class__.__name__
            if model_name == 'LogisticRegression':
                coefficients = pd.Series(m['model'].coef_.flatten(), index=m['features'])
                coefficients = coefficients[coefficients != 0.0]
                n_features_selected.append(len(coefficients))
                for f in coefficients.index:
                    if f not in feature_names_selected:
                        feature_names_selected.append(f)
            elif model_name == 'DummyClassifier':
                n_features_selected.append(0)

        self.avg_n_features_selected = np.mean(n_features_selected)
        self.median_n_features_selected = np.median(n_features_selected)
        self.sum_n_features_selected = np.sum(n_features_selected)

        # calculate the unique features used in the inner nodes of dt
        dt_features = self.dt.feature_names_in_[np.where(self.dt.feature_importances_ > 0.0)]
        self.dt_features_used = len(dt_features)


    def predict(self, X):
        leaf_idx = self.dt.apply(X)

        pred = np.zeros(len(X))
        for idx in np.unique(self.leaf_idx):
            X_leaf = X[leaf_idx == idx]

            # if there are any samples from the test set in this leaf node
            if len(X_leaf) > 0:
                model = self.lm_dict[idx]['model']
                features = self.lm_dict[idx]['features']
                pred[leaf_idx == idx] = model.predict(X_leaf[features])

        return pred


    def predict_proba(self, X):
        leaf_idx = self.dt.apply(X)

        pred = np.zeros((len(X)))
        for idx in np.unique(self.leaf_idx):
            X_leaf = X[leaf_idx == idx]

            # if there are any samples from the test set in this leaf node
            if len(X_leaf) > 0:
                model = self.lm_dict[idx]['model']
                features = self.lm_dict[idx]['features']
                pred[leaf_idx == idx] = model.predict_proba(X_leaf[features])[:, 1]

        return pred

    def get_log_loss_per_leaf(self, X_test, y_test):
        leaf_idx = self.dt.apply(X_test)

        loss_per_leaf = []
        for idx in np.unique(self.leaf_idx):
            X_leaf = X_test[leaf_idx == idx]
            pred = self.lm_dict[idx]['model'].predict_proba(X_leaf)

            loss_per_leaf.append([idx, log_loss(y_test[leaf_idx == idx], pred[:, 1], labels=[0, 1]), len(X_leaf)])

        self.loss_per_leaf = pd.DataFrame(loss_per_leaf, columns=['idx', 'loss', 'samples'])

    def plot(self, show_n=5):
        for leaf_index in np.unique(self.leaf_idx):
            leaf_model_data = self.lm_dict[leaf_index]
            if isinstance(leaf_model_data['model'], LogisticRegression):
                coefficients = leaf_model_data['model'].coef_.flatten()
                feature_names = leaf_model_data['features']
                n_features = min(len(feature_names), show_n)

                # Sort indices based on absolute value of coefficients
                sorted_indices = np.argsort(np.abs(coefficients))[-n_features:]

                # Re-arrange coefficients and feature_names based on sorted indices
                sorted_coefficients = [coefficients[i] for i in sorted_indices]
                sorted_feature_names = [feature_names[i] for i in sorted_indices]

                # Reverse them to have the highest absolute coefficients first
                sorted_coefficients = sorted_coefficients[::-1]
                sorted_feature_names = sorted_feature_names[::-1]

                fig, axes = plt.subplots(1, n_features, figsize=(14 * (n_features / 5), 6))


                for i, (coefficient, feature_name) in enumerate(zip(sorted_coefficients, sorted_feature_names), start=0):

                    max_x_value = leaf_model_data['max_x_values'][feature_name]
                    min_x_value = leaf_model_data['min_x_values'][feature_name]
                    x_values = np.linspace(min_x_value, max_x_value, 100)  # Replace with actual min and max if available
                    y_impact = coefficient * x_values  # Compute the generalized impact on y

                    sns.lineplot(
                        x=x_values,
                        y=y_impact,
                        ax=axes[i],
                        linewidth=2,
                        color="darkblue"
                    )

                    # Add a title for each subplot based on the feature name
                    axes[i].set_title(feature_name, fontsize=14)

                    # Add grid to the subplot
                    axes[i].grid(True)

                    # Add a horizontal line at y=0 with dotted gray style
                    axes[i].axhline(0, color='gray', linestyle='--')

                plt.subplots_adjust(wspace=0.4)
                fig.suptitle(
                    f"Leaf {leaf_index}: Effect of Features on Target",
                    fontsize=18,
                )
                fig.tight_layout(rect=[0, 0.03, 1, 0.90])
            else:
                pass

            plt.show()

    def extract_decision_rules(self, X_feature_names, verbose=0, output_filename='decision_rules.csv'):
        tree = self.dt.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold

        # Recursively traverse the tree to build the decision rules for each leaf
        def recurse(node, depth, decision_path):
            if children_left[node] == children_right[node]:  # leaf
                # Print the decision rule that leads to this leaf
                return [" AND ".join(decision_path)]
            else:
                # Name of the feature for this node
                feature_name = X_feature_names[feature[node]]

                # Left child
                left_rule = f"{feature_name} <= {threshold[node]:.2f}"
                left_rules = recurse(children_left[node], depth + 1, decision_path + [left_rule])

                # Right child
                right_rule = f"{feature_name} > {threshold[node]:.2f}"
                right_rules = recurse(children_right[node], depth + 1, decision_path + [right_rule])

                return left_rules + right_rules

        # Start the recursion with the root node and an empty decision path
        decision_rules = recurse(0, 0, [])

        if verbose == 1:
            # Print the decision rules in a formatted way
            for i, rule in enumerate(decision_rules):
                print(f"Segment {i + 1}: {rule}")

        segments = ['Segment {}'.format(i + 1) for i, _ in enumerate(decision_rules)]

        rules = [segment.split(" AND ") for segment in decision_rules]

        # Create DataFrame
        df = pd.DataFrame(index=segments)

        # Assign conditions to the dataframe, expanding into multiple columns
        for i, condition in enumerate(rules):
            for j, decision_rule in enumerate(condition):
                df.loc[segments[i], j] = decision_rule

        df.columns = ['Rule {}'.format(i + 1) for i in range(df.shape[1])]

        # Create feature columns starting at start_col
        for feature_name in X_feature_names:
            df[feature_name] = np.nan  # Initialize columns with NaNs

        for segment_i, (_, segment_lr_model) in enumerate(self.lm_dict.items()):
            model = segment_lr_model['model']
            if isinstance(model, DummyClassifier):
                continue
            coefficients = model.coef_[0]
            intercept = model.intercept_[0]
            features = segment_lr_model['features']

            # Place the intercept in the specified column
            df.loc[segments[segment_i], 'Intercept'] = f"{intercept:.3f}"

            # Fill the corresponding feature columns with coefficients
            for feature, coef in zip(features, coefficients):
                df.loc[segments[segment_i], feature] = f"{coef:.3f}"

        # Now, anonymize the feature names
        anonymized_feature_names = {orig_name: f"X{index+1}" for index, orig_name in enumerate(X_feature_names)}
        df = df.rename(columns=anonymized_feature_names)

        # Remove the unused feature columns
        df.dropna(axis=1, how='all', inplace=True)


        df = df.replace('\n', ' ', regex=True).replace('  ', ' ', regex=True)

        # Save to CSV
        df.to_csv(output_filename)
        print(f'Decision rules have been written to {output_filename}')


class LLM_Regressor:
    def __init__(self, max_depth=4, ccp_alpha=0.0, random_state=42, lr_alpha=1.0, min_leaf_size=1, ffs=False):
        self.max_depth = max_depth
        self.random_state = random_state
        self.min_leaf_size = min_leaf_size
        self.lr_alpha = lr_alpha
        self.dt = DecisionTreeRegressor(max_depth=max_depth, ccp_alpha=ccp_alpha, min_samples_leaf=self.min_leaf_size,
                                        random_state=random_state)
        self.predictors = []  # intialize after decision tree has predicted

    def fit(self, X, y):
        self.dt.fit(X, y)
        labels = self.dt.apply(X)
        if np.min(labels) > 0:  # sometimes labels are indexed with 1, should not usually happen
            labels -= np.min(labels)  # scales back to 0
        for pred_iterator, i in enumerate(np.unique(labels)):
            self.predictors.append(Ridge(random_state=self.random_state + i, alpha=self.lr_alpha))
            indexes = np.where(labels == i)[0]
            if len(indexes) > 0:
                curr_cluster_X, curr_cluster_y = X.iloc[indexes], y.iloc[indexes]  # i+1, since labels start with 1
                self.predictors[pred_iterator].fit(curr_cluster_X, curr_cluster_y)

    def predict(self, X):
        labels = self.dt.apply(X)
        prediction = np.zeros(X.shape[0])
        preds = 0
        if np.min(labels) > 0:  # sometimes labels are indexed with 1, should not usually happen
            labels -= np.min(labels)  # scales back to 0
        for pred_iterator, i in enumerate(np.unique(labels)):
            indices = np.where(labels == i)[0]
            if len(indices) > 0:
                curr_cluster_X = X.iloc[indices]
                prediction[indices] = self.predictors[pred_iterator].predict(curr_cluster_X)
                preds += len(indices)
        if preds != len(X):
            print("Not all X are predicted")  # just for testing, remove later
        return prediction


class DummyClassifier(nn.Module):
    # set to torch module
    def __init__(self, pred_class):
        super().__init__()
        self.pred_class = int(pred_class)
        self.pred_vec = [0, 0]
        self.pred_vec[self.pred_class] = 1

    def forward(self, X):
        return torch.tensor(self.pred_class, dtype=torch.float32, device=X.device)

    def predict(self, X):
        return [self.pred_class] * len(X)

    def predict_proba(self, X):
        return np.array([self.pred_vec] * len(X))