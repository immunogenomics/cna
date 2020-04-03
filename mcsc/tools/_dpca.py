import numpy as np
from sklearn.utils.extmath import randomized_svd
import scipy.stats as st
from ._stats import conditional_permutation, type1errors
import time

def std(X):
    std = X.std(axis=0); std[std < 1e-3] = np.nan
    return X / std

def stdconf(X, confounders):
    X = X - confounders.dot(
        np.linalg.solve(confounders.T.dot(confounders), confounders.T.dot(X)))
    return std(X)

def stdconf_cov(X, confounders):
    X = stdconf(X, confounders)
    return X.T.dot(X) / len(X)

def stdconf_score(X, A, confounders, absv=True):
    X = stdconf(X, confounders)
    if absv:
        return np.abs(X.dot(A))
    else:
        return X.dot(A)

def calc_covs(X, C, s):
    if s is None:
        s = np.ones((len(X), 1))
    else:
        s = np.hstack([s, np.ones((len(X), 1))])

    indices = np.concatenate([[0], np.cumsum(C)])
    return np.array([
            stdconf_cov(X[i:j], s[i:j])
            for i, j in zip(indices[:-1], indices[1:])
        ])

def calc_scores(X, A, C, s, absv=True):
    if s is None:
        s = np.ones((len(X), 1))
    else:
        s = np.hstack([s, np.ones((len(X), 1))])

    indices = np.concatenate([[0], np.cumsum(C)])
    return np.array([
            stdconf_score(X[i:j], A, s[i:j], absv=absv)
            for i, j in zip(indices[:-1], indices[1:])
        ])

def calc_means(X, C):
    indices = np.concatenate([[0], np.cumsum(C)])
    return np.array([
            X[i:j].mean(axis=0)
            for i, j in zip(indices[:-1], indices[1:])
        ])

def regress_means(means, Y, T):
    YY = means
    if T is None:
        XX = np.hstack([Y.reshape((-1,1)), np.ones((len(Y), 1))]).astype(np.float64)
    else:
        XX = np.hstack([Y.reshape((-1,1)), T, np.ones((len(Y), 1))]).astype(np.float64)

    XTXi = np.linalg.inv(XX.T.dot(XX))
    Beta = XTXi.dot(XX.T.dot(YY))

    return Beta[0]

def regress_covs(covs, Y, T):
    YY = np.array([cov.flatten() for cov in covs])
    if T is None:
        XX = np.hstack([Y.reshape((-1,1)), np.ones((len(Y), 1))]).astype(np.float64)
    else:
        XX = np.hstack([Y.reshape((-1,1)), T, np.ones((len(Y), 1))]).astype(np.float64)

    XTXi = np.linalg.inv(XX.T.dot(XX))
    Beta = XTXi.dot(XX.T.dot(YY))

    dim = int(np.sqrt(len(Beta.T)))
    return Beta[0].reshape((dim, dim))

def d1m(X, Y, C, B=None, T=None,
        Nnull=10000, seed=0):
    np.random.seed(seed)

    means = calc_means(X, C)
    Delta = regress_means(means, Y, T)

    nulldist = []
    nullY = conditional_permutation(B, Y.astype(np.float64), Nnull)

    for i, shuff in enumerate(nullY.T):
        if i % 500 == 0: print(i, end='.')
        Delta_ = regress_means(means, shuff, T)
        nulldist.append(Delta_)
    print()
    nulldist = np.array(nulldist)

    p = (nulldist**2 >= Delta**2).sum(axis=0) / len(nulldist)
    p[p == 0] = 0.5/Nnull
    nullp = (np.array([np.argsort(np.argsort(x)) for x in nulldist.T**2]) + 1) / len(nulldist)
    fwer, fep95, fdr = type1errors(-np.log10(p), -np.log10(nullp))

    return Delta, p, fwer, fep95, fdr

def d2m(X, Y, C, B=None, T=None, s=None,
        Nnull=10000, seed=0):
    np.random.seed(seed)

    covs = calc_covs(X, C, s)
    Delta = regress_covs(covs, Y, T)
    lowertri = np.tril_indices(len(Delta), k=-1)
    Delta = Delta[lowertri] # get lower triangle of Delta

    nulldist = []
    nullY = conditional_permutation(B, Y.astype(np.float64), Nnull)

    for i, shuff in enumerate(nullY.T):
        if i % 500 == 0: print(i, end='.')
        Delta_ = regress_covs(covs, shuff, T)
        nulldist.append(Delta_[lowertri])
    print()
    nulldist = np.array(nulldist)

    p = (nulldist**2 >= Delta**2).sum(axis=0) / len(nulldist)
    p[p==0] = 0.5/Nnull
    nullp = (np.array([np.argsort(np.argsort(x)) for x in nulldist.T**2]) + 1) / len(nulldist)
    fwer, fep95, fdr = type1errors(-np.log10(p), -np.log10(nullp))

    return Delta, p, fwer, fep95, fdr

def dpca(X, Y, C, B=None, T=None, s=None,
        n_components=5,
        Nnull=10000, seed=0):
    np.random.seed(seed)

    covs = calc_covs(X, C, s)
    Delta = regress_covs(covs, Y, T)
    U, svs, dpcXfeature = randomized_svd(Delta, n_components=n_components)
    # U and dpcXfeature are only identical up to signs of components, so that
    #   svs is always non-negative. this is why we only have to do a one-sided
    #   test below.

    nulldist = []
    nullY = conditional_permutation(B, Y.astype(np.float64), Nnull)

    for i, shuff in enumerate(nullY.T):
        if i % 500 == 0: print(i, end='.')
        Delta_ = regress_covs(covs, shuff, T)
        _, svs_, _ = randomized_svd(Delta_, n_components=n_components)
        nulldist.append(svs_)
    print()
    nulldist = np.array(nulldist)

    p = (nulldist >= svs).sum(axis=0) / len(nulldist)
    p[p==0] = 0.5/Nnull
    #nullp = (np.array([np.argsort(np.argsort(x)) for x in nulldist.T]) + 1) / len(nulldist)
    #fwer, fep95, fdr = type1errors(-np.log10(p), -np.log10(nullp))
    fwer, fep95, fdr = type1errors(svs, nulldist.T)

    return Delta, dpcXfeature, svs, p, fwer, fep95, fdr

