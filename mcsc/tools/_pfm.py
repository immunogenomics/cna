import numpy as np
import pandas as pd
import scipy.stats as st
from ._stats import conditional_permutation, empirical_fdrs, \
    empirical_fwers, minfwer_loo, numtests, numtests_loo
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
import time, gc

# creates a neighborhood frequency matrix
#   requires data.uns[sampleXmeta][ncellsid] to contain the number of cells in each sample.
#   this can be obtained using mcsc.pp.sample_size
def nfm(data, nsteps=3, sampleXmeta='sampleXmeta', ncellsid='C', key_added='sampleXnh'):
    a = data.uns['neighbors']['connectivities']
    C = data.uns[sampleXmeta][ncellsid].values
    colsums = np.array(a.sum(axis=0)).flatten() + 1
    s = np.repeat(np.eye(len(C)), C, axis=0)

    for i in range(nsteps):
        print(i)
        s = a.dot(s/colsums[:,None]) + s/colsums[:,None]
    snorm = s / C

    data.uns[key_added] = snorm.T

# creates a cluster frequency matrix
#   data.obs[clusters] must contain the cluster assignment for each cell
def cfm(data, clusters, sampleXmeta='sampleXmeta', sampleid='id', key_added=None):
    if key_added is None:
        key_added = 'sampleX'+clusters

    sm = data.uns[sampleXmeta]
    nclusters = len(data.obs[clusters].unique())
    cols = []
    for i in range(nclusters):
        cols.append(clusters+'_'+str(i))
        sm[cols[-1]] = data.obs.groupby(sampleid)[clusters].aggregate(
            lambda x: (x.astype(np.int)==i).mean())

    data.uns[key_added] = sm[cols].values
    sm.drop(columns=cols, inplace=True)

def pca(data, repname='sampleXnh', npcs=None):
    if npcs is None:
        npcs = min(*data.uns[repname].shape)
    s = data.uns[repname].copy()
    s = s - s.mean(axis=0)
    s = s / s.std(axis=0)
    ssT = s.dot(s.T)
    V, d, VT = np.linalg.svd(ssT)
    U = s.T.dot(V) / np.sqrt(d)
    del s; gc.collect()

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

    # residualize confounders out of X and Y
    resid = np.eye(len(Y)) - T.dot(np.linalg.solve(T.T.dot(T), T.T))
    Y = resid.dot(Y)
    X = resid.dot(X)

    # get null
    NY = conditional_permutation(B, Y.astype(np.float64), Nnull).T

    return X, Y, NY

def linreg(data, Y, B, T, npcs=50, L=0, repname='sampleXnh', Nnull=500, newrep=None):
    if npcs is None:
        npcs = data.uns[repname].shape[1] - 1
    X = data.uns[repname]
    X, Y, NY = prepare(B, T, X, Y, Nnull)

    if newrep is None:
        newrep = repname+'.resid'
    data.uns[newrep] = X
    pca(data, repname=newrep, npcs=npcs)
    X = data.uns[newrep+'_sampleXpc']
    sqevs = data.uns[newrep+'_sqevals']
    X *= np.sqrt(sqevs)

    # compute mse
    G = np.linalg.solve(X.T.dot(X) + L*np.eye(len(X.T)), X.T)
    H = X.dot(G)
    beta = G.dot(Y)
    Yhat = H.dot(Y)
    mse = ((Y-Yhat)**2).mean() / Y.var()
    msemarg = ((Y[:,None] - X*beta)**2).mean(axis=0) / Y.var()

    # null testing
    nulls = []
    nullbeta2s = []
    for Y_ in NY:
        beta_ = G.dot(Y_)
        Yhat_ = H.dot(Y_)
        mse_ = ((Y_-Yhat_)**2).mean() / Y_.var()
        nulls.append(mse_)

        msemarg_ = ((Y_[:,None] - X*beta_)**2).mean(axis=0) / Y_.var()
        nullbeta2s.append(msemarg_)
    nulls = np.array(nulls)
    nullbeta2s = np.array(nullbeta2s)
    p = ((nulls <= mse).sum() + 1) / (len(nulls)+1)
    betap = ((nullbeta2s <= msemarg).sum(axis=0) + 1) / (len(nullbeta2s)+1)

    return p, beta, betap

def mixedmodel(data, Y, B, T, npcs=50, repname='sampleXnh', usepca=True):
    if npcs is None:
        npcs = data.uns[repname].shape[1] - 1
    if usepca and repname+'_sampleXpc' not in data.uns.keys():
        pca(data, repname=repname, npcs=npcs)

    if usepca:
        X = data.uns[repname+'_sampleXpc'][:,:npcs]
        sqevs = data.uns[repname+'_sqevals'][:npcs]
        X *= np.sqrt(sqevs)
    else:
        X = data.uns[repname]

    pcnames = ['PC'+str(i) for i in range(len(X.T))]
    covnames = ['T'+str(i) for i in range(len(T.T))]

    df = pd.DataFrame(
        np.hstack([X, T, B.reshape((-1,1)), Y.reshape((-1,1))]),
        columns=pcnames+covnames+['batch', 'Y'])
    md1 = smf.mixedlm(
        'Y ~ ' + '+'.join(covnames+pcnames),
        df, groups='batch')
    mdf1 = md1.fit(reml=False)
    print(mdf1.summary())
    md0 = smf.mixedlm(
        'Y ~ ' + '+'.join(covnames),
        df, groups='batch')
    mdf0 = md0.fit(reml=False)
    print(mdf0.summary())

    llr = mdf1.llf - mdf0.llf
    p = st.chi2.sf(2*llr, len(pcnames))

    return p, None, None
