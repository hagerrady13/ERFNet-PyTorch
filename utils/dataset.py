import numpy as np

def calc_dataset_stats(dataset, axis=0, ep=1e-7):
    return (np.mean(dataset, axis=axis) / 255.0).tolist(), (
        np.std(dataset + ep, axis=axis) / 255.0).tolist()