import numpy as np
import scipy.stats as st

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

def tail_counts(z, znull, atol=1e-8, rtol=1e-5):
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
    bins = np.concatenate([z2[ix] - atol - rtol*z2[ix], [np.inf]])
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
    """
    znull is assumed to be of shape len(z) x k, where k is the number of
        null instantiations.
    """
    # get tail counts
    if znull.shape[0] != len(z):
        print('ERROR: znull is shape', znull.shape, 'and z is shape', z.shape)
    tails = tail_counts(z, znull)
    ranks = len(z) - np.argsort(np.argsort(z**2))

    # compute FWERs
    fwer = ((tails > 0).sum(axis=0) + 1) / (znull.shape[1] + 1)

    # compute FDPs
    fdp = tails / ranks
    fdr = fdp.mean(axis=0)
    fep95 = np.percentile(fdp, 95, axis=0, interpolation='higher')

    return fwer, fep95, fdr

def significant_fwer(z, znull, level, atol=1e-8, rtol=1e-5):
    level = np.array(level)
    maxs = np.max(znull**2, axis=0)
    thresholds = np.percentile(
        maxs,
        100*(1-level),
        interpolation='higher')

    if len(thresholds.shape) == 0:
        return z**2 - thresholds > (atol + rtol*thresholds) # analogous to np.allclose
    else:
        return np.array([
            z**2 - thresh > (atol + rtol*thresh) # analogous to np.allclose
            for thresh in thresholds]).T

#def significant_fwer_loo(z, level, atol=1e-8, rtol=1e-5):
#    level = np.array(level)
#    maxs = np.max(z**2, axis=0)
#
#    ix = np.arange(len(maxs))
#    loo = np.array([
#        maxs[ix != i]
#        for i in ix
#        ])
#    thresholds = np.percentile(loo, 100*(1-level), axis=1, interpolation='higher')
#
#    if len(thresholds.shape) <= 1:
#        return z**2 - thresholds > (atol + rtol*thresholds) # analogous to np.allclose
#    else:
#        return np.array([
#            z**2 - thresh > (atol + rtol*thresh) # analogous to np.allclose
#            for thresh in thresholds]).T

def num_indep_tests(z, fwer):
    q = 0.05
    if (fwer <= q).sum() > 0:
        p = np.ones(len(z))
        p[fwer <= q] = st.chi2.sf(z[z**2 >= z2thresh], 1)

        return (fwer[fwer <= q] * p[fwer <= q]).sum() / (p[fwer <= q]**2).sum()
    else:
        return None
