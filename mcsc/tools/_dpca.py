import numpy as np
from sklearn.decomposition import TruncatedSVD
import scipy.stats as st
from ._stats import conditional_permutation
import time

def std(X):
    std = X.std(axis=0); std[std < 1e-3] = np.nan
    return X / std

def stdcov(X, confounders):
    X = X - confounders.dot(
        np.linalg.solve(confounders.T.dot(confounders), confounders.T.dot(X)))
    X = std(X)
    return X.T.dot(X) / (len(X)-1)

def calc_covs(X, C, s):
    if s is None:
        s = np.ones((len(X), 1))
    else:
        s = np.hstack([s, np.ones((len(X), 1))])

    indices = np.concatenate([[0], np.cumsum(C)])
    return np.array([
            stdcov(X[i:j], s[i:j])
            for i, j in zip(indices[:-1], indices[1:])
        ])

def regression(covs, Y, T):
    YY = np.array([cov.flatten() for cov in covs])
    if T is None:
        XX = np.hstack([Y.reshape((-1,1)), np.ones((len(Y), 1))]).astype(np.float64)
    else:
        XX = np.hstack([Y.reshape((-1,1)), T, np.ones((len(Y), 1))]).astype(np.float64)

    XTXi = np.linalg.inv(XX.T.dot(XX))
    Beta = XTXi.dot(XX.T.dot(YY))

    dim = int(np.sqrt(len(Beta.T)))
    return Beta[0].reshape((dim, dim))

def dpca(X, Y, C, B=None, T=None, s=None,
        n_components=5,
        significance=0.05,
        Nnull=10000, seed=0,
        outdetail=1):
    np.random.seed(seed)

    covs = calc_covs(X, C, s)
    Delta = regression(covs, Y, T)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(Delta)
    svs = svd.singular_values_
    dpcsXfeatures = svd.components_

    nulldist = []
    nullY = conditional_permutation(B, Y.astype(np.float64), Nnull)

    for i, shuff in enumerate(nullY.T):
        if i % 500 == 0: print(i, end='.')
        Delta_ = regression(covs, shuff, T)
        svd.fit(Delta_)
        nulldist.append(svd.singular_values_)
    nulldist = np.array(nulldist)
    pvals = (nulldist >= svs).sum(axis=0) / len(nulldist)

    #Delta = pd.DataFrame(Delta, index=features, columns=features)
    return Delta, dpcsXfeatures, svs, pvals

