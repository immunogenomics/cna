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

def pca(data, repname='sampleXnh', npcs=None):
    if npcs is None:
        npcs = min(*data.uns[repname].shape)
    s = data.uns[repname].copy()
    s = s - s.mean(axis=0)
    s = s / s.std(axis=0)
    ssT = s.dot(s.T)
    V, d, VT = np.linalg.svd(ssT)
    U = s.T.dot(V) / np.sqrt(d)

    data.uns[repname+'_sqevals'] = d[:npcs]
    data.uns[repname+'_featureXpc'] = U[:,:npcs]
    data.uns[repname+'_sampleXpc'] = V[:,:npcs]

def prepare(B, T, X, Y, Nnull):
    B_oh = np.array([
        B == b for b in np.unique(B)
        ]).T.astype(np.float)
    if T is None:
        T = B_oh
    else:
        T = np.hstack([T, B_oh])

    # get null
    NY = conditional_permutation(B, Y.astype(np.float64), Nnull).T

    # residualize confounders out of X and Y
    resid = np.eye(len(Y)) - T.dot(np.linalg.solve(T.T.dot(T), T.T))

    return resid.dot(X), resid.dot(Y), resid.dot(NY.T).T

def linreg(data, Y, B, T, npcs=50, repname='sampleXnh', Nnull=500):
    if npcs is None:
        npcs = data.uns[repname].shape[1] - 1
    X = data.uns[repname]
    X, Y, NY = prepare(B, T, X, Y, Nnull)

    data.uns['temp'] = X
    pca(data, repname='temp', npcs=npcs)
    X = data.uns['temp_sampleXpc']
    sqevs = data.uns['temp_sqevals']

    # compute mse
    beta = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
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
    p = ((nulls <= mse).sum() + 1) / (len(nulls)+1)

    del data.uns['temp']
    return p, beta, sqevs

def pcridgereg(data, Y, B, T, L=1e6, repname='sampleXnh', Nnull=500,
    returnbeta=False):
    X = data.uns[repname+'_sampleXpc']
    N, M = X.shape
    sqevs = data.uns[repname+'_sqevals']
    X = X * np.sqrt(sqevs) * np.sqrt(N)
    X, Y, NY = prepare(B, T, X, Y, Nnull)

    # compute mse
    H = X.dot(np.linalg.solve(X.T.dot(X) + N*L*np.eye(M), X.T))
    Yhat = H.dot(Y)
    mse = ((Y-Yhat)**2).mean()

    if returnbeta:
        return np.linalg.solve(X.T.dot(X) + N*L*np.eye(M), X.T.dot(Y)), Yhat, mse

    # null testing
    Yhat_ = H.dot(NY.T)
    nulls = ((Yhat_ - NY.T)**2).mean(axis=0)
    p = ((nulls <= mse).sum() + 1) / (len(nulls)+1)
    return p

def kernelridgereg(data, Y, B, T, L=1, repname='sampleXnh', Nnull=500):
    X = data.uns[repname]
    N = len(X)
    X = X * np.sqrt(N)
    X, Y, NY = prepare(B, T, X, Y, Nnull)

    # compute MSE
    K = X.dot(X.T)
    H = K.dot(np.linalg.inv(K + L*N*np.eye(N)))
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

def marg_minp(data, Y, B, T, nfeatures=20, repname='sampleXnh_sampleXpc', Nnull=500):
    if nfeatures is None:
        nfeatures = data.uns[repname].shape[1]
    X = data.uns[repname][:,:nfeatures]
    X, Y, NY = prepare(B, T, X, Y, Nnull)

    # compute stats
    beta2 = (X.T.dot(Y) / (X**2).sum(axis=0))**2

    # null testing
    nulls = []
    for Y_ in NY:
        beta2_ = (X.T.dot(Y_) / (X**2).sum(axis=0))**2
        nulls.append(beta2_)
    nulls = np.array(nulls)

    ps = ((nulls >= beta2).sum(axis=0) + 1) / (len(nulls)+1)
    return ps.min() * len(ps)
