import time
import scipy.stats as st
import numpy as np

def prepare(a, B, C, s, T, Y): # see analyze(..) for parameter descriptions
    # add dummy batch info if none supplied
    if B is None:
        B = np.ones(len(Y))

    # verify samples are sorted by batch (for null permutation)
    if any(np.diff(B) < 0):
        print('ERROR: samples must be sorted by batch')

    # set all sample sizes to be identical if no information supplied
    if C is None:
        C = np.ones(len(Y))

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

def conditional_permutation(B, Y, num): # assumes B is sorted and that Y and B line up
    starts = np.concatenate([
        [0],
        np.where(B[:-1] != B[1:])[0] + 1,
        [len(B)]])
    ix = np.concatenate([
        np.argsort(np.random.randn(j-i, num), axis=0) + i
        for i, j in zip(starts[:-1], starts[1:])
        ])
    return Y[ix]

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

def analyze(a, Y, C=None, B=None, T=None, s=None,
        maxsteps=20, loops=1,
        Nnull=100, seed=0,
        outdetail=1, outfreq=1):
    """
    Carries out multi-condition analysis.

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
    Nnull (int): number of null permutations to use to estimate mean and variance of
        null distribution
    seed (int): random seed to use
    outdetail (int): level of printed output detail
    outfreq (int): how often to print output (every outfreq steps)

    Returns:
    2d array: a set of maxsteps+1 arrays, each of which gives the diffusion score of
        each cell at a given timepoint of the diffusion
    2d array: a set of maxsteps+1 arrays, each of which gives the z-score of each cell
        at a given timepoint of the diffusion
    1d array: a set of maxsteps+1 floats, each of which gives the squared z-score
        threshold for a 5% family-wise error rate accounting for correlated tests
    2d array: a set of maxsteps+1 arrays, each of which gives the standard deviation of
        diffusion scores under the null for each cell at a given timepoint in the
        diffusion
    2d array: a set of maxsteps+1 arrays, each of which gives an unadjusted p-value
        for each cell at a given timepoint in the diffusion
    """
    def corr(g,h):
        return (g - g.mean()).dot(h - h.mean()) / (len(g)*g.std()*h.std())

    np.random.seed(seed)

    a, B, u, w, y = prepare(a, B, C, s, T, Y)
    
    D = np.zeros((maxsteps+1, len(y)))
    z = np.zeros((maxsteps+1, len(y)))
    bonf_z2 = np.zeros(maxsteps+1)
    stds = np.zeros((maxsteps+1, len(y)))
    mlp = np.ones((maxsteps+1, len(y)))

    colsums = np.array(a.sum(axis=0)).flatten() + loops
    nullmean = get_null_mean(B, C, u, w, Y)

    D[0] = y - nullmean
    Dnull = get_null(B, C, u, w, Y, Nnull) - nullmean[:,None]

    t0 = time.time()
    for i in range(1, maxsteps+1):
        if outdetail > 0 and i % outfreq == 0:
            print('step {:d} ({:.1f}s)'.format(i, time.time()-t0))
        
        D[i] = a.dot(D[i-1] / colsums) + loops * D[i-1]/colsums
        Dnull = a.dot(Dnull / colsums[:,None]) + loops * Dnull / colsums[:,None]
        stds[i] = np.sqrt((Dnull**2).mean(axis=1))

        z[i] = D[i] / stds[i]
        bonf_z2[i] = np.percentile(np.max((Dnull / stds[i][:,None])**2, axis=0), 95)

        if outdetail > 1 and i % outfreq == 0:
            print(
                '\t{:d} significant cells'.format(
                    (z[i]**2 > bonf_z2[i]).sum()))
        if outdetail > 2 and i % outfreq == 0:
            print(
                '\tR2 {:.4f}'.format(
                    corr(D[i-1], D[i])**2))

        mlp[i] = -(st.norm.logsf(np.abs(z[i])/np.log(10) + np.log10(2)))

    return D, z, bonf_z2, stds, mlp