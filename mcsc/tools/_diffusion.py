import numpy as np
import scipy.stats as st
from ._stats import conditional_permutation
import time

def prepare(a, B, C, s, T, Y): # see analyze(..) for parameter descriptions
    # add dummy batch info if none supplied
    if B is None:
        B = np.ones(len(Y))

    # verify samples are sorted by batch (for null permutation)
    if any(np.diff(B) < 0):
        print('ERROR: samples must be sorted by batch')

    # translate sample-level outcome to cell-level outcome
    y = np.repeat(1000*Y, C).astype(np.float64)

    # add all-ones column to sample-level covariates
    if T is not None:
        T_ = np.hstack([T, np.ones((len(T), 1))])
    else:
        T_ = np.ones((len(Y), 1))

    # combine sample-level and cell-level covariates into one cell-level matrix
    # and create cell-level weight vector
    if s is not None:
        u = np.hstack([
            np.repeat(T_, C, axis=0),
            s]).astype(np.float64)
    else:
        u = np.repeat(T_, C, axis=0)
    w = np.repeat(C, C).astype(np.float64)

    # project covariates out of outcome and weight it
    y = y - u.dot(np.linalg.solve(u.T.dot(u / w[:,None]), u.T.dot(y / w)))
    y /= w

    return a, B, u, w, y

def get_null(B, C, u, w, Y, num):
    nullY = conditional_permutation(B, Y.astype(np.float64), num)
    nully = np.repeat(1000*nullY, C, axis=0)
    nully -= u.dot(np.linalg.solve(u.T.dot(u / w[:,None]), u.T.dot(nully / w[:,None])))
    nully /= w[:,None]

    return nully

def get_null_mean(B, C, u, w, Y):
    nullMeans = np.array([
        np.mean(Y[B==b])
        for b in B
    ])
    nullmeans = np.repeat(1000*nullMeans, C, axis=0)
    nullmeans -= u.dot(np.linalg.solve(u.T.dot(u / w[:,None]), u.T.dot(nullmeans / w)))
    nullmeans /= w

    return nullmeans

def diffusion(a, Y, C, B=None, T=None, s=None,
        maxsteps=100, loops=1,
        keepevery=None,
        stopthresh=0.9999,
        significance=0.05,
        Nnull=100, seed=0,
        outdetail=1, outfreq=5):
    """
    Carries out multi-condition analysis using diffusion.

    Parameters:
    a (scipy.sparse.csr.csr_matrix): adjacency matrix of graph, assumed not to contain
        self loops
    Y (1d numpy array): sample-level outcome to associate to graph location
    C (1d numpy array): number of cells in each sample
    B (1d numpy array): batch id of each sample
    T (2d numpy array): sample-level covariates to adjust for
    s (2d numpy array): cell-level covariates to adjust for
    maxsteps (int): maximum number of steps to take in random walk
    loops (float): strength of self loops to add
    keepevery (int): default (None) causes diffusion to return only the last timestep. Any
        other int will cause diffusion to return every keepevery-th timestep.
    stopthresh (float): diffusion stops when correlation between successive sets of z-scores
        goes above this number. Passing None causes diffusion to continue for maxsteps steps.
    significance (float or iterable of floats): set of significance levels for which
        diffusion will supply z-score thresholds accounting for correlation among cells.
    Nnull (int): number of null permutations to use to estimate mean and variance of
        null distribution
    seed (int): random seed to use
    outdetail (int): level of printed output detail
    outfreq (int): how often to print output (every outfreq steps)

    Returns:
    1d/2d array: a set of arrays, each of which gives the z-score of each cell at a given
        saved timepoint of the diffusion
    0d/1d/2d array: a set of floats, each of which gives the squared z-score threshold for a
        the family-wise error rate set by significance (default 0.05), accounting for
        correlated tests. Dimensionality depends on whether >1 timepoints are saved and on
        dimensionality of significance.
    1d/2d array: a set of arrays, each of which gives an unadjusted p-value for each cell at
        a given saved timepoint in the diffusion
    1d/2d array: a set of arrays, each of which gives the diffusion score of each cell
        at a given saved imepoint of the diffusion
    0d/1d array: the set of timepoints corresponding to the returned results
    """
    def corr(g,h):
        gstd = g.std()
        hstd = h.std()
        if np.isclose([gstd, hstd], [0,0]).any():
            return 0
        else:
            return (g - g.mean()).dot(h - h.mean()) / (len(g)*gstd*hstd)
    def save_snapshot():
        ts.append(t)
        ds.append(dcurr)
        zs.append(zcurr)
        bonf_z2s.append(np.percentile(
            np.max((dnull / stdcurr[:,None])**2, axis=0),
            100*(1-significance)))
    def stop_condition():
        if stopthresh is None:
            return False
        else:
            return corr(zcurr, zprev) > stopthresh

    # initialize random seed and result variables
    np.random.seed(seed)
    ds = list()
    zs = list()
    bonf_z2s = list()
    ts = list()

    # prepare data
    a, B, u, w, y = prepare(a, B, C, s, T, Y)
    colsums = np.array(a.sum(axis=0)).flatten() + loops
    nullmean = get_null_mean(B, C, u, w, Y)
    significance = np.array(significance)

    # initialize time 0 of random walk
    dcurr = y - nullmean
    zcurr = np.zeros(len(dcurr))
    dnull = get_null(B, C, u, w, Y, Nnull) - nullmean[:,None]

    # do random walk
    start = time.time()
    print('t=0')
    for t in range(1, maxsteps+1):
        # take step
        dprev, zprev = dcurr, zcurr
        dcurr = a.dot(dcurr / colsums) + loops * dcurr/colsums
        dnull = a.dot(dnull / colsums[:,None]) + loops * dnull / colsums[:,None]

        # compute z-scores
        stdcurr = np.sqrt((dnull**2).mean(axis=1))
        zcurr = dcurr / stdcurr

        # decide whether to stop and whether to save current timestep
        if stop_condition():
            save_snapshot()
            break
        if keepevery is not None and (t-1) % keepevery == 0:
            save_snapshot()

        # print progress
        if outdetail > 0 and t % outfreq == 0:
            print('t={:d} ({:.1f}s)'.format(t, time.time()-start))
    # save last timepoint if it isn't already saved
    print('t={:d}: finished'.format(t))
    if t not in ts:
        save_snapshot()

    # compute p-values and return
    zs = np.array(zs).squeeze()
    mlps = -st.norm.logsf(
        np.abs(zs)/np.log(10) + np.log10(2)
        )

    return zs, \
        np.array(bonf_z2s).squeeze(), \
        mlps, \
        np.array(ds).squeeze(), \
        np.array(ts).squeeze() \
