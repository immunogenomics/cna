import numpy as np
import scipy.stats as st

def conditional_permutation(B, Y, num):
    """
    Permutes Y conditioned on B num different times.
    """
    batchind = np.array([
        np.where(B == b)[0] for b in np.unique(B)
        ])
    ix = np.concatenate([
        bi[np.argsort(np.random.randn(len(bi), num), axis=0)]
        for bi in batchind
        ])
    bix = np.zeros((len(Y), num)).astype(np.int)
    bix[np.concatenate(batchind)] = ix
    result = Y[bix]
    return Y[bix]

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

def empirical_fdrs(z, znull, thresholds):
    """
    znull is assumed to be of shape len(z) x k, where k is the number of
        null instantiations.
    """
    # get tail counts
    if znull.shape[0] != len(z):
        print('ERROR: znull is shape', znull.shape, 'and z is shape', z.shape)
    tails = tail_counts(thresholds, znull)
    ranks = tail_counts(thresholds, z)

    # compute FWER (superceded by empirical_fwers)
    #fwer = ((tails > 0).sum(axis=0) + 1) / (znull.shape[1] + 1)

    # compute FDPs
    fdp = tails / ranks
    fdr = fdp.mean(axis=0)
    #fep95 = np.percentile(fdp, 95, axis=0, interpolation='higher')

    return fdr

def empirical_fwers(z, Nmaxz2, atol=1e-8, rtol=1e-5):
    # Nmaxz2 is assumed to be of length k where k is number of null simulates
    tc = tail_counts(z, np.sqrt(Nmaxz2), atol=atol, rtol=rtol)[0]
    return (tc + 1)/(len(Nmaxz2)+1)

def minfwer_loo(Nmaxz2, atol=1e-8, rtol=1e-5):
    tc = np.array([(Nmaxz2 >= z2).sum() for z2 in Nmaxz2])
    return (tc + 1)/len(Nmaxz2)

def numtests(Nmaxz2):
    j, k = 0, 10
    maxs = np.sort(Nmaxz2)[::-1]
    fwers = (np.arange(j, k)+1)/(len(maxs)+1)
    ps = st.chi2.sf(maxs[j:k], 1)
    return 1/(ps.dot(fwers) / fwers.dot(fwers))

def numtests_loo(Nmaxz2):
    return np.array([
        numtests(Nmaxz2[np.arange(len(Nmaxz2)) != i])
        for i in range(len(Nmaxz2))
        ])
