import numpy as np
import scipy.stats as st
from ._stats import conditional_permutation, empirical_fdrs, \
    empirical_fwers, minfwer_loo, numtests, numtests_loo
import time, gc

def nns(data, ms=3, sampleXmeta='sampleXmeta', key_added='sampleXnh'):
    a = data.uns['neighbors']['connectivities']
    C = data.uns[sampleXmeta].C.values
    colsums = np.array(a.sum(axis=0)).flatten() + 1
    s = np.repeat(np.eye(len(C)), C, axis=0)

    for i in range(ms):
        if i % 1 == 0:
            print(i)
        s = a.dot(s/colsums[:,None]) + s/colsums[:,None]
    snorm = s / C

    data.uns[key_added] = snorm.T

def pca(data, repname='sampleXnh'):
    s = data.uns[repname].copy()
    s = s - s.mean(axis=0)
    s = s / s.std(axis=0)
    ssT = s.dot(s.T)
    V, d, VT = np.linalg.svd(ssT)
    U = s.T.dot(V) / np.sqrt(d)

    data.uns[repname+'_sqevals'] = d
    data.obsm[repname+'_featureXpc'] = U
    data.uns[repname+'_sampleXpc'] = V

def linreg(data, Y, B, T, nfeatures=20, repname='sampleXnh_sampleXpc', Nnull=500):
    B_oh = np.array([
        B == b for b in np.unique(B)
        ]).T.astype(np.float)
    if nfeatures is None:
        nfeatures = data.uns[repname].shape[1]
    X = data.uns[repname][:,:nfeatures]

    if T is None:
        T = B_oh
    else:
        T = np.hstack([T, B_oh])

    # get null
    NY = conditional_permutation(B, Y.astype(np.float64), Nnull).T

    # residualize confounders out of X and Y
    resid = np.eye(len(Y)) - T.dot(np.linalg.solve(T.T.dot(T), T.T))
    Y = resid.dot(Y)
    X = resid.dot(X)
    NY = resid.dot(NY.T).T

    # compute mse
    H = X.dot(np.linalg.solve(X.T.dot(X), X.T))
    Yhat = H.dot(Y)
    mse = ((Y-Yhat)**2).mean()

    # null testing
    nulls = []
    for Y_ in NY:
        Yhat_ = H.dot(Y_)
        mse_ = ((Y_-Yhat_)**2).mean()
        nulls.append(mse_)
    nulls = np.array(nulls)

    return ((nulls <= mse).sum() + 1) / (len(nulls)+1)
