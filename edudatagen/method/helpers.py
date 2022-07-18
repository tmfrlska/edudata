import numpy as np
from scipy.stats import mode, iqr


def proper(X_df=None, y_df=None, random_state=None):
    sample_indicies = y_df.sample(frac=1, replace=True, random_state=random_state).index
    y_df = y_df.loc[sample_indicies]

    if X_df is None:
        return y_df

    else:
        X_df = X_df.loc[sample_indicies]
        return X_df, y_df


def smooth(dtype, y_synth, y_real_min, y_real_max, random_state=None):
    indices = [True for _ in range(len(y_synth))]

    y_synth_mode = mode(y_synth)
    if y_synth_mode.count / len(y_synth) > 0.7:
        indices = np.logical_and(indices, y_synth != y_synth_mode.mode)

    # exclude from smoothing if data are top-coded - approximate check
    y_synth_sorted = np.sort(y_synth)
    top_coded = 10 * np.abs(y_synth_sorted[-2]) < np.abs(y_synth_sorted[-1]) - np.abs(y_synth_sorted[-2])
    if top_coded:
        indices = np.logical_and(indices, y_synth != y_real_max)

    bw = 0.9 * len(y_synth[indices]) ** -1/5 * np.minimum(np.std(y_synth[indices]), iqr(y_synth[indices]) / 1.34)
    np.random.seed(random_state)
    y_synth[indices] = np.array([np.random.normal(loc=value, scale=bw) for value in y_synth[indices]])
    if not top_coded:
        y_real_max += bw
    y_synth[indices] = np.clip(y_synth[indices], y_real_min, y_real_max)
    if dtype == 'int':
        y_synth[indices] = y_synth[indices].astype(int)

    return y_synth
