import numpy as np
import pandas as pd
import warnings
import scipy.stats as st
import anndata
import gc
from packaging import version
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

def _df_to_array(data, x):
    if type(x) in [pd.DataFrame, pd.Series]:
        if all(x.index == data.samplem.index):
            return x.values
        else:
            print('ERROR: index does not match index of data.samplem')
    else:
        return x

# creates a neighborhood abundance matrix
def _nam(data, nsteps=None, maxnsteps=15, self_weight=1, show_progress=False):
    out = select_output(show_progress)
    
    def R(A, B):
        return ((A - A.mean(axis=0))*(B - B.mean(axis=0))).mean(axis=0) \
            / A.std(axis=0) / B.std(axis=0)

    S = pd.get_dummies(data.obs_sampleids)[data.samplem.index.values]
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
    snorm.index.name = data.samplem.index.name
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

    return NAM[:, keep], keep

# residualizes covariates and batch information out of NAM
def _resid_nam(NAM, covs, batches, ridge=None, show_progress=False):
    out = select_output(show_progress)
    
    N = len(NAM)
    NAM_ = NAM - NAM.mean(axis=0)
    if covs is None:
        covs = np.ones((N, 0))
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

    return NAM_ / NAM_.std(axis=0), M, len(C.T)

#performs SVD of NAM
def _svd_nam(NAM):
    U, svs, UT = np.linalg.svd(NAM.dot(NAM.T))
    V = NAM.T.dot(U) / np.sqrt(svs)

    return (U, svs, V)

def nam(data, batches=None, covs=None, filter_samples=None,
    nsteps=None, self_weight=1, max_frac_pcs=0.15, suffix='', ks = None,
    force_recompute=False, show_progress=False, **kwargs):
    out = select_output(show_progress)

    def safe_same(A, B):
        if A is None: A = np.zeros(0)
        if B is None: B = np.zeros(0)
        if A.shape == B.shape:
            return np.allclose(A, B, equal_nan = True)
        else:
            return False
    
    # error checking
    covs = _df_to_array(data, covs)
    batches = _df_to_array(data, batches)
    if batches is None:
        batches = np.array([1]*data.N)
    if filter_samples is None:
        if covs is not None:
            filter_samples = ~np.any(np.isnan(covs), axis=1)
        else:
            filter_samples = np.repeat(True, data.N)
    else:
        filter_samples = _df_to_array(data, filter_samples)

    du = data.uns
    # compute and QC NAM
    if force_recompute or \
        'NAM.T'+suffix not in du or \
        not safe_same(batches, du['_batches'+suffix]):
        print('qcd NAM not found; computing and saving', file=out)
        NAM = _nam(data, nsteps=nsteps, self_weight=self_weight, show_progress=show_progress)
        NAMqc, keep = _qc_nam(NAM.values, batches, show_progress=show_progress)
        du['NAM.T'+suffix] = pd.DataFrame(NAMqc, index=NAM.index, columns=NAM.columns[keep]).T
        du['keptcells'+suffix] = keep
        du['_batches'+suffix] = batches

    # correct for batch/covariates and SVD
    if force_recompute or \
        'NAM_sampleXpc'+suffix not in du or \
        not safe_same(covs, du['_covs'+suffix]) or \
        not safe_same(filter_samples, du['_filter_samples'+suffix]):

        print('covariate-adjusted NAM not found; computing and saving', file=out)
        du['_filter_samples'+suffix] = filter_samples
        NAM = du['NAM.T'+suffix].T.iloc[filter_samples]
        NAM_resid, M, r = _resid_nam(NAM.values,
                                covs[filter_samples] if covs is not None else covs,
                                batches[filter_samples] if batches is not None else batches,
                                show_progress=show_progress)

        print('computing SVD', file=out)
        du['NAM_resid.T'+suffix] = NAM_resid.T
        U, svs, V = _svd_nam(NAM_resid)
        npcs = min(V.shape[1], max([10]+[int(max_frac_pcs * data.N)]+[ks if ks is not None else []][0]))
        du['NAM_sampleXpc'+suffix] = pd.DataFrame(U,
            index=NAM.index,
            columns=['PC'+str(i) for i in range(1, len(U.T)+1)])
        du['NAM_svs'+suffix] = svs
        du['NAM_varexp'+suffix] = svs / len(U) / len(V)
        du['NAM_nbhdXpc'+suffix] = pd.DataFrame(V[:,:npcs],
            index=NAM.columns,
            columns=['PC'+str(i) for i in range(1, npcs+1)])
        du['_M'+suffix] = M
        du['_r'+suffix] = r
        du['_covs'+suffix] = (np.zeros(0) if covs is None else covs)
