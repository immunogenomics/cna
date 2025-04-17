import numpy as np
import pandas as pd
import warnings
import scipy.stats as st
import anndata
import gc
from packaging import version
from argparse import Namespace
from ._out import select_output

def get_connectivity(data):
    av = anndata.__version__
    if type(av) == str:
        av = version.parse(av)
    if av < version.parse("0.7.2"):
        return data.uns["neighbors"]["connectivities"]
    else:
        return data.obsp["connectivities"]
        
def diffuse_stepwise(data, s, maxnsteps=15, show_progress=False, self_weight=1):
    out = select_output(show_progress)
    
    # find connectivity matrix
    a = get_connectivity(data)

    # normalize
    colsums = np.array(a.sum(axis=0)).flatten() + self_weight

    # do diffusion
    for i in range(maxnsteps):
        print('\ttaking step', i+1, file=out)
        s = a.dot(s/colsums[:,None]) + self_weight*s/colsums[:,None]
        yield s

def diffuse(data, s, nsteps, show_progress=False, self_weight=1):
    for s in diffuse_stepwise(data, s, maxnsteps=nsteps,
                              show_progress=show_progress,
                              self_weight=self_weight):
        pass
    return s

# creates a neighborhood abundance matrix
def _nam(data, sid_name, sids=None, nsteps=None, maxnsteps=15, self_weight=1, show_progress=False):
    out = select_output(show_progress)
    
    def R(A, B):
        return ((A - A.mean(axis=0))*(B - B.mean(axis=0))).mean(axis=0) \
            / A.std(axis=0) / B.std(axis=0)

    S = pd.get_dummies(data.obs[sid_name])
    if sids is not None:
        S = S[sids]
    C = S.sum(axis=0)

    prevmedkurt = np.inf
    old_s = np.zeros(S.shape)
    for i, s in enumerate(diffuse_stepwise(data, S, self_weight=self_weight, maxnsteps=maxnsteps)):
        medkurt = np.median(st.kurtosis(s/C, axis=1))
        R2 = R(s, old_s)**2
        old_s = s
        print('\tmedian kurtosis:', medkurt+3, file=out)
        print('\t20th percentile R2(t,t-1):', np.percentile(R2, 20), file=out)
        if nsteps is None:
            if prevmedkurt - medkurt < 3 and i+1 >= 3:
                print('stopping after', i+1, 'steps', file=out)
                break
            prevmedkurt = medkurt
        elif i+1 == nsteps:
            break
        gc.collect()

    snorm = (s / C).T
    snorm.index.name = sid_name
    snorm.index = snorm.index.astype(str)
    return snorm

def _batch_kurtosis(NAM, batches):
    return st.kurtosis(np.array([
                NAM[batches == b].mean(axis=0) for b in np.unique(batches)
                ]), axis=0) + 3

#qcs a NAM to remove neighborhoods that are batchy
def _qc_nam(NAM, batches, show_progress=False):
    out = select_output(show_progress)
    
    N = len(NAM)
    if len(np.unique(batches)) == 1:
        keep = np.repeat(True, len(NAM.T))
        return NAM, keep

    kurtoses = _batch_kurtosis(NAM, batches)
    threshold = max(6, 2*np.median(kurtoses))
    print('throwing out neighborhoods with batch kurtosis >=', threshold, file=out)
    keep = (kurtoses < threshold)
    print('keeping', keep.sum(), 'neighborhoods', file=out)

    return NAM.iloc[:, keep], keep

#performs SVD of NAM
def svd_nam(NAM):
    NAM = NAM - NAM.mean(axis=0)
    NAM = NAM / NAM.std(axis=0)
    U, svs, UT = np.linalg.svd(NAM.dot(NAM.T))
    V = NAM.T.dot(U) / np.sqrt(svs)

    return (pd.DataFrame(U,
                        index=NAM.index,
                        columns=['PC'+str(i) for i in range(1, len(U.T)+1)]),
            pd.Series(svs, index=['PC'+str(i) for i in range(1, len(U.T)+1)]),
            pd.DataFrame(V.values,
                        index=NAM.columns,
                        columns=['PC'+str(i) for i in range(1, len(U.T)+1)])
    )

# residualizes covariates and batch information out of NAM
def _resid_nam(NAM, covs, batches, ridge=None, npcs=None, show_progress=False):
    out = select_output(show_progress)
    
    N = len(NAM)
    NAM_ = NAM - NAM.mean(axis=0)
    if covs is None:
        covs = pd.DataFrame(np.ones((N, 0)), index=NAM.index)
    else:
        covs = (covs - covs.mean(axis=0))/covs.std(axis=0)

    if batches is None or len(np.unique(batches)) == 1:
        C = covs
        if len(C.T) == 0:
            M = np.eye(N)
        else:
            M = np.eye(N) - C.dot(np.linalg.solve(C.T.dot(C), C.T))
        NAM_ = M.dot(NAM_)
    else:
        B = pd.get_dummies(batches).values
        B = (B - B.mean(axis=0))/B.std(axis=0)
        C = np.hstack([B, covs])

        if ridge is not None:
            ridges = [ridge]
        else:
            ridges = [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 0]

        for ridge in ridges:
            L = np.diag([1]*len(B.T)+[0]*(len(C.T)-len(B.T)))
            M = np.eye(N) - C.dot(np.linalg.solve(C.T.dot(C) + ridge*len(C)*L, C.T))
            NAM_ = M.dot(NAM_)

            kurtoses = _batch_kurtosis(NAM_, batches)

            print('\twith ridge', ridge, 'median batch kurtosis = ',
                    np.median(kurtoses), file=out)

            if np.median(kurtoses) <= 6:
                break

    # standardize NAM
    NAM_ = NAM_ / NAM_.std(axis=0)
    NAM_ = pd.DataFrame(NAM_, index=NAM.index, columns=NAM.columns)

    # do SVD
    U, svs, V = svd_nam(NAM_)
    if npcs is None:
        npcs = len(V.T)

    # store results
    res = Namespace()
    res.M = M
    res.r = len(C.T)
    res.namresid = NAM_
    res.namresid_sampleXpc = U
    res.namresid_nbhdXpc = V
    res.namresid_svs = svs[:npcs]
    res.namresid_varexp = svs / len(U) / len(V)

    return res

def nam(data, sid_name, batches=None,
    nsteps=None, self_weight=1, max_frac_pcs=0.15, suffix='', ks = None,
    show_progress=False, **kwargs):
    out = select_output(show_progress)
    
    # ensure batches are properly formatted and initialized
    if batches is None:
        batches = pd.Series(np.ones(len(data.obs[sid_name].unique())), index=data.obs[sid_name].unique())
    
    # compute and QC NAM
    print('computing NAM', file=out)
    NAM = _nam(data, sid_name, nsteps=nsteps, self_weight=self_weight, show_progress=show_progress)
    NAMqc, keep = _qc_nam(NAM, batches, show_progress=show_progress)

    return pd.DataFrame(NAMqc, index=NAM.index, columns=NAM.columns[keep], dtype=float), keep