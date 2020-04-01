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

def tail_counts(z, znull):
    """
    Computes the number of null z-scores of equal or greater magnitude than each supplied
        z-score, for each null instantiation.

    znull is assumed to be either of shape len(z) or len(z) x k, where k is the number of
        null instantiations.
    """
    # re-shape znull if only one set of null results is supplied
    if len(znull.shape) == 1:
        znull = znull.reshape((-1, 1))

    # square z-scores and prepare to sort/un-sort them
    z2 = z**2
    ix = np.argsort(z2); iix = np.argsort(ix)

    # ask numpy to make a histogram with sorted squared z-scores as boundaries
    bins = np.concatenate([z2[ix], [np.inf]])
    hist = np.array([
            np.histogram(zn2, bins=bins)[0]
        for zn2 in znull.T**2])

    # convert histogram into tail counts (in-place)
    tails = np.flip(hist, axis=1)
    np.cumsum(tails, axis=1, out=tails)
    tails = np.flip(tails, axis=1)

    # return tail counts for un-sorted z-scores
    return tails[:, iix]

def type1errors(z, znull):
    # get tail counts
    tails = tail_counts(z, znull)
    ranks = len(z) - np.argsort(np.argsort(z**2))

    # compute FWERs
    fwer = (tails > 0).mean(axis=0)

    # compute FDPs
    fdp = tails / ranks
    fdr = fdp.mean(axis=0)
    fep95 = np.percentile(fdp, 95, axis=0)

    return fwer, fep95, fdr

def significant_fwer(z, znull, level):
    level = np.array(level)
    thresholds = np.percentile(
        np.max(znull**2, axis=0),
        100*(1-level))

    if len(thresholds.shape) == 0:
        return z**2 >= thresholds
    else:
        return np.array([
            z**2 >= thresh
            for thresh in thresholds]).T



















