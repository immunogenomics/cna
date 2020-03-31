import numpy as np

def conditional_permutation(B, Y, num):
    """
    Permutes Y conditioned on B num different times. Note: this function assumes that B is
    sorted.
    """
    starts = np.concatenate([
        [0],
        np.where(B[:-1] != B[1:])[0] + 1,
        [len(B)]])
    ix = np.concatenate([
        np.argsort(np.random.randn(j-i, num), axis=0) + i
        for i, j in zip(starts[:-1], starts[1:])
        ])
    return Y[ix]
