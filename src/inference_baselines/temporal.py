"""The source-attributed current + mean-of-five representation."""
import numpy as np


def current_mean5(corpus, positions=None):
    index = np.asarray(corpus.index if positions is None else corpus.index[positions])
    result = np.empty((len(index), 288), np.float32)
    for file_id in np.unique(index[:, 0]):
        mask = index[:, 0] == file_id
        edges = index[mask, 1]
        if (edges < 4).any():
            raise ValueError("Five-vector features require four preceding rows")
        values = corpus.x[file_id]
        total = np.zeros((len(edges), 144), np.float64)
        for lag in (4, 3, 2, 1, 0):
            total += values[edges-lag]
        result[mask, :144] = values[edges]
        result[mask, 144:] = total/5
    return result
