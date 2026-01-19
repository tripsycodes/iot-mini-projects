import numpy as np
import pandas as pd

def time_aware_ewma(series, time_diffs, tau=15):
    ewma_values = [series.iloc[0]]
    for t in range(1, len(series)):
        alpha = 1 - np.exp(-time_diffs[t-1] / tau)
        ewma_values.append(alpha * series.iloc[t] + (1 - alpha) * ewma_values[-1])
    return pd.Series(ewma_values, index=series.index)
