import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from edudata.method import Method, proper, smooth
# global variables
from edudata import NUM_COLS_DTYPES, CAT_COLS_DTYPES


class RandomforestMethod(Method):
    def __init__(self, dtype, smoothing=False, proper=False, n_estimators=30, max_depth=10, random_state=None, *args, **kwargs):
        self.dtype = dtype
        self.smoothing = smoothing
        self.proper = proper
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        if self.dtype in CAT_COLS_DTYPES:
            self.rf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
        if self.dtype in NUM_COLS_DTYPES:
            self.rf = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)

    def fit(self, X_df, y_df):
        if self.proper:
            X_df, y_df = proper(X_df=X_df, y_df=y_df, random_state=self.random_state)

        X_df, y_df = self.prepare_dfs(X_df=X_df, y_df=y_df, normalise_num_cols=False, one_hot_cat_cols=True)
        if self.dtype in NUM_COLS_DTYPES:
            self.y_real_min, self.y_real_max = np.min(y_df), np.max(y_df)

        X = X_df.to_numpy()
        y = y_df.to_numpy()
        self.rf.fit(X, y)

        # save the y distribution wrt trained tree nodes
        leaves = [Counter(i).most_common(n=1)[0][0] for i in self.rf.apply(X)]
        leaves_y_df = pd.DataFrame({'leaves': leaves, 'y': y})
        self.leaves_y_dict = leaves_y_df.groupby('leaves').apply(lambda x: x.to_numpy()[:, -1]).to_dict()

    def predict(self, X_test_df):
        np.random.seed(self.random_state)
        X_test_df, _ = self.prepare_dfs(X_df=X_test_df, normalise_num_cols=False, one_hot_cat_cols=True, fit=False)
        X_test = X_test_df.to_numpy()
        leaves_pred = [Counter(i).most_common(n=1)[0][0] for i in self.rf.apply(X_test)]
        y_pred = np.zeros(len(leaves_pred), dtype=object)

        leaves_pred_index_df = pd.DataFrame({'leaves_pred': leaves_pred, 'index': range(len(leaves_pred))})
        leaves_pred_index_dict = leaves_pred_index_df.groupby('leaves_pred').apply(lambda x: x.to_numpy()[:, -1]).to_dict()
        for leaf, indices in leaves_pred_index_dict.items():
            if leaf in self.leaves_y_dict.keys():
                y_pred[indices] = np.random.choice(self.leaves_y_dict[leaf], size=len(indices), replace=True)
            else:
                origin_leaves = list(self.leaves_y_dict.keys())
                temp = origin_leaves[np.searchsorted(origin_leaves, leaf)-1]
                y_pred[indices] = np.random.choice(self.leaves_y_dict[temp], size=len(indices), replace=True)

        if self.smoothing and self.dtype in NUM_COLS_DTYPES:
            y_pred = smooth(self.dtype, y_pred, self.y_real_min, self.y_real_max, random_state=self.random_state)

        return y_pred
