import numpy as np
import scipy.stats as st
from ._stats import conditional_permutation, type1errors, \
    empirical_fwers, minfwer_loo, numtests, numtests_loo
import time

def prepare(B, C, s, T, Y): # see analyze(..) for parameter descriptions
    # add dummy batch info if none supplied
    if B is None:
        B = np.ones(len(Y))

    # verify samples are sorted by batch (for null permutation)
    if any(np.diff(B) < 0):
        print('ERROR: samples must be sorted by batch')

    # add batch indicators to sample-level covariates
    batchvalues = np.unique(B)
    batchdesign = np.array([
        [b == j for j in batchvalues]
        for b in B]).astype(np.float64)
    if T is not None:
        T_ = np.hstack([T, batchdesign])
    else:
        T_ = batchdesign

    # combine sample-level and cell-level covariates into one cell-level matrix
    # and create cell-level weight vector
    if s is not None:
        u = np.hstack([
            np.repeat(T_, C, axis=0),
            s]).astype(np.float64)
    else:
        u = np.repeat(T_, C, axis=0)
    w = np.repeat(C, C).astype(np.float64)

    # translate sample-level outcome to cell-level outcome
    y = np.repeat(1000*Y, C).astype(np.float64)

    # project covariates out of outcome and weight it
    y = y - u.dot(np.linalg.solve(u.T.dot(u / w[:,None]), u.T.dot(y / w)))
    y /= w

    return B, u, w, y

def get_null(B, C, u, w, Y, num):
    nullY = conditional_permutation(B, Y.astype(np.float64), num)
    nully = np.repeat(1000*nullY, C, axis=0)
    nully -= u.dot(np.linalg.solve(u.T.dot(u / w[:,None]), u.T.dot(nully / w[:,None])))
    nully /= w[:,None]

    return nully

def diffusion_minfwer(a, Y, C, B=None, T=None, s=None,
        diffusion=True,
        maxsteps=50, loops=1,
        Nnull=100, seed=0,
        outdetail=1, outfreq=5):
    if seed is not None:
        np.random.seed(seed)

    # prepare data
    B, u, w, y = prepare(B, C, s, T, Y)
    colsums = np.array(a.sum(axis=0)).flatten() + loops

    # initialize time 0 of random walk. Naming conventions:
    #   d = raw diffusion scores, z = z-scores, h = number of hits
    #   *_c = current, *_p = previous, N* = null
    d_c = y
    Nd_c = get_null(B, C, u, w, Y, Nnull)
    z = list()#np.zeros((maxsteps+1, len(d_c)))
    Nmaxz2 = list()#np.zeros((maxsteps+1, Nnull))

    # do diffusion
    t = 0
    start = time.time()
    if outdetail > 0: print('t = 0')
    for t in range(1, maxsteps+1):
        # take step
        if diffusion:
            d_c = a.dot(d_c / colsums) + loops * d_c/colsums
            Nd_c = a.dot(Nd_c / colsums[:,None]) + loops * Nd_c / colsums[:,None]
        else:
            d_c = a.dot(d_c) / colsums + loops * d_c/colsums
            Nd_c = a.dot(Nd_c) / colsums[:,None] + loops * Nd_c / colsums[:,None]

        # compute z-scores
        # compute standard deviations under null for each cell
        Nd_c2 = Nd_c**2
        Nd_c2sum = Nd_c2.sum(axis=1)
        std_c = np.sqrt(Nd_c2sum / Nnull)
        std_c_loo = np.sqrt(
            (Nd_c2sum[:,None] - Nd_c2) / (Nnull-1)
            )
        # compute z-scores
        z.append(d_c / std_c)
        Nmaxz2.append(((Nd_c / std_c_loo)**2).max(axis=0))

        # print progress
        if outdetail > 0 and t % outfreq == 0:
            print('t={:d} ({:.1f}s)'.format(t, time.time()-start))
    if outdetail > 0: print('t={:d}: finished'.format(t))

    z = np.array(z)
    Nmaxz2 = np.array(Nmaxz2)

    ntests = np.array([numtests(Nmaxz2_) for Nmaxz2_ in Nmaxz2])
    Nntests = np.array([numtests_loo(Nmaxz2_) for Nmaxz2_ in Nmaxz2])
    t_f = np.argmin(st.chi2.sf((z**2).max(axis=1), 1) * ntests)
    z_f = z[t_f]

    Nt_f = np.argmin(
            st.chi2.sf(Nmaxz2, 1) * Nntests,
            axis=0)
    Nz2_f = np.array([Nmaxz2[Nt_f[i], i] for i in range(Nnull)])
    fwer = empirical_fwers(z[t_f], Nz2_f)

    if outdetail > 0:
        print('max z2:', (z[t_f]**2).max())
        print('min p:', st.chi2.sf((z[t_f]**2).max(), 1))
        print('min padj:', ntests[t_f]*st.chi2.sf((z[t_f]**2).max(), 1))
        print('min fwer:', fwer.min())

    return z[t_f], \
        fwer, \
        numtests(Nz2_f), \
        t_f + 1, \
        Nt_f

def diffusion_expgrowth(a, Y, C, B=None, T=None, s=None,
        diffusion=True,
        maxsteps=50, loops=1,
        growthreq=0.05, nontrivial=100,
        keepevery=None,
        significance=None,
        Nnull=100, seed=0,
        outdetail=1, outfreq=5):
    def process_step():
        # compute standard deviations under null for each cell
        Nd_c2 = Nd_c**2
        Nd_c2sum = Nd_c2.sum(axis=1)
        std_c = np.sqrt(Nd_c2sum / Nnull)
        std_c_loo = np.sqrt(
            (Nd_c2sum[:,None] - Nd_c2) / (Nnull-1)
            )
        # compute z-scores
        z_c = d_c / std_c
        Nz_c = Nd_c / std_c_loo
        # compute number of hits
        Nmaxz2 = (Nz_c**2).max(axis=0)
        fwer = empirical_fwers(z_c, Nmaxz2)
        h_c = (fwer <= 0.05).sum()
        return z_c, Nz_c, h_c
    def stop_condition():
        # the +1 below avoids a numpy warning without meaningfully changing results
        if growthreq is None:
            return False
        else:
            return h_p >= nontrivial and (h_c-h_p+1) / (h_p+1) < growthreq
    #TODO: significance shouldn't be performed inside save_snapshot
    def save_snapshot():
        ts.append(t)
        ds.append(d_c)
        zs.append(z_c)
        if significance is None:
            #fwer, fep95, fdr = type1errors(z_c, Nz_c)
            Nmaxz2 = (Nz_c**2).max(axis=0)
            fwer, fep95, fdr = empirical_fwers(z_c, Nmaxz2), None, None

            ntests.append(numtests(Nmaxz2))
            #fwer[fwer <= 0.02] = ntests*st.chi2.sf(z_c[fwer <= 0.02]**2, 1)
            fwers.append(fwer)
            feps95.append(fep95)
            fdrs.append(fdr)
        else:
            hits = significant_fwer(z_c, Nz_c, significance)
            fwers.append(hits)

    if seed is not None:
        np.random.seed(seed)
    if significance is not None:
        significance = np.array(significance)

    # prepare data
    B, u, w, y = prepare(B, C, s, T, Y)
    colsums = np.array(a.sum(axis=0)).flatten() + loops

    # initialize results
    ts, ds, zs, fwers, feps95, fdrs = list(), list(), list(), list(), list(), list()
    ntests = list()

    # initialize time 0 of random walk. Naming conventions:
    #   d = raw diffusion scores, z = z-scores, h = number of hits
    #   *_c = current, *_p = previous, N* = null
    d_c = y
    Nd_c = get_null(B, C, u, w, Y, Nnull)
    z_c = np.zeros(d_c.shape)
    Nz_c = np.zeros(Nd_c.shape)
    h_c, h_p = 0, 0

    # do diffusion
    t = 0
    start = time.time()
    if outdetail > 0: print('t = 0')
    if keepevery is not None:
        save_snapshot()
    for t in range(1, maxsteps+1):
        # take step
        if diffusion:
            d_c = a.dot(d_c / colsums) + loops * d_c/colsums
            Nd_c = a.dot(Nd_c / colsums[:,None]) + loops * Nd_c / colsums[:,None]
        else:
            d_c = a.dot(d_c) / colsums + loops * d_c/colsums
            Nd_c = a.dot(Nd_c) / colsums[:,None] + loops * Nd_c / colsums[:,None]

        # compute z-scores
        h_p = h_c
        z_c, Nz_c, h_c = process_step()

        # print progress
        if outdetail > 0 and t % outfreq == 0:
            print('t={:d} ({:.1f}s)'.format(t, time.time()-start))
        if outdetail > 1:
            ntests = num_indep_tests_fast(Nz_c)
            zmax = np.max(z_c**2)
            pmin = st.chi2.sf(zmax, 1)
            padj = pmin*ntests
            print('\t{} hits, {:.2f} rel growth, max z2 {:.1f} ({:.2e}) (({:.2e}))'.format(
                h_c, (h_c-h_p+1)/(h_p+1),
                zmax,
                pmin,
                padj
                ))

        # decide whether to stop and whether to save current timestep
        if keepevery is not None and t % keepevery == 0:
            save_snapshot()
        if stop_condition():
            break
    if outdetail > 0: print('t={:d}: finished'.format(t))

    # save last timepoint if it isn't already saved
    if t not in ts:
        save_snapshot()

    if outdetail > 0:
        print('max z2:', (z_c**2).max())
        print('min p:', st.chi2.sf((z_c**2).max(), 1))
        print('min padj:', fwers[-1].min())

    return np.squeeze(np.array(zs)), \
        np.squeeze(np.array(fwers)), \
        np.squeeze(np.array(feps95)), \
        np.squeeze(np.array(fdrs)), \
        np.squeeze(np.array(ntests)), \
        np.squeeze(np.array(ts))


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
    growthreq (float): diffusion stops when percent growth of number of hits goes below this
        number, provided number of hits is above the value contained in nontrivial. Passing
        None causes diffusion to continue for maxsteps steps.
    keepevery (int): z scores, FWERs, and FEPs will be saved every keepevery steps and
        returned. Default is None, in which case only the results at the last time step are
        returned.
    significance (float): if this is not None, then the method will return an indicator vector
        containing for each cell whether it has FWER < the supplied threshold. This is faster.
    Nnull (int): number of null permutations to use to estimate mean and variance of
        null distribution and to perform FWER/FEP/FDR control.
    seed (int): random seed to use
    outdetail (int): level of printed output detail
    outfreq (int): how often to print output (every outfreq steps)

    Returns:
    1d/2d array: a set of arrays, each of which gives the z-score of each cell at a given
        saved timepoint of the diffusion
    1d/2d array: a set of arrays, each of which gives the estimated FWER of each cell at a
        given saved timepoint of the diffusion. If significance is not None, then this instead
        contains True/False values indicating whether each cell is significant at the FWER
        theshold in sigificance.
    1d/2d array: a set of arrays, each of which gives the estimated probability of FDP <= 5%
        of each cell at a given saved timepoint of the diffusion. If significance is None,
        this array will be empty
    1d/2d array: a set of arrays, each of which gives the estimated FDR (= average FDP) of
        each cell at a given saved timepoint of the diffusion. If significance is None, this
        array will be empty
    0d/1d array: the set of timepoints corresponding to the returned results
    """

def diffusion():
    pass
