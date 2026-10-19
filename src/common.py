import numpy as np

def rank(score, tiebreak):
    return np.lexsort((np.asarray(tiebreak), -np.asarray(score)))

def apfd(failed_ranked):
    f = np.asarray(failed_ranked, bool)
    n, pos = f.size, np.flatnonzero(f) + 1
    return 1 - pos.sum() / (n * pos.size) + 1 / (2 * n)

def dfe_update(P, failed_ids, alpha=0.8):
    P *= 1 - alpha
    P[failed_ids] += alpha
