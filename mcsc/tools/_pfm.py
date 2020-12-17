import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
import time, gc
from argparse import Namespace
import mcsc.tools._stats as stats

###### CNAv2/v3
# creates a neighborhood abundance matrix
#   requires data.uns[sampleXmeta][ncellsid] to contain the number of cells in each sample.
#   this can be obtained using mcsc.pp.sample_size
def nam(data, nsteps=None, sampleXmeta='sampleXmeta', ncellsid='C'):
    a = data.uns['neighbors']['connectivities']
    C = data.uns[sampleXmeta][ncellsid].values
    colsums = np.array(a.sum(axis=0)).flatten() + 1
    s = np.repeat(np.eye(len(C)), C, axis=0)

    prevmedkurt = np.inf
    for i in range(15):
        print('taking step', i+1)
        s = a.dot(s/colsums[:,None]) + s/colsums[:,None]

        if nsteps is None:
            medkurt = np.median(st.kurtosis(s/C, axis=1))
            print('median excess kurtosis:', medkurt)
            if prevmedkurt - medkurt < 3:
                print('stopping after', i+1, 'steps')
                break
            prevmedkurt = medkurt
        elif i+1 == nsteps:
            break

    snorm = s / C
    return snorm.T

def _qc(NAM, batches):
    N = len(NAM)
    if len(np.unique(batches)) == 1:
        print('warning: only one unique batch supplied to qc')
        keep = np.repeat(True, len(NAM.T))
        return NAM, keep

    B = pd.get_dummies(batches).values
    B = (B - B.mean(axis=0))/B.std(axis=0)

    batchcorr = B.T.dot(NAM - NAM.mean(axis=0)) / N / NAM.std(axis=0)
    batchcorr = np.nan_to_num(batchcorr) # if batch is constant then 0 correlation
    maxbatchcorr2 = np.max(batchcorr**2, axis=0)
    print('throwing out neighborhoods with maxbatchcorr2 >=', 2*np.median(maxbatchcorr2))
    keep = (maxbatchcorr2 < 2*np.median(maxbatchcorr2))
    print('keeping', keep.sum(), 'neighborhoods')

    return NAM[:, keep], keep

def _prep(NAM, covs, batches, ridge=None):
    N = len(NAM)
    NAM_ = NAM - NAM.mean(axis=0)
    if covs is None:
        covs = np.ones((N, 0))
    else:
        covs = (covs - covs.mean(axis=0))/covs.std(axis=0)

    if len(np.unique(batches)) == 1:
        print('warning: only one unique batch supplied to prep')
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

            batchcorr = B.T.dot(NAM_ - NAM_.mean(axis=0)) / len(B) / NAM_.std(axis=0)
            maxbatchcorr = np.max(batchcorr**2, axis=0)

            print('with ridge', ridge, 'median max sq batch correlation =',
                    np.percentile(maxbatchcorr, 50))

            if np.percentile(maxbatchcorr, 50) <= 0.025:
                break

    NAM_ = NAM_ / NAM_.std(axis=0)

    U, sv, UT = np.linalg.svd(NAM_.dot(NAM_.T))
    V = NAM_.T.dot(U) / np.sqrt(sv)

    return (U, sv, V), M, len(C.T)

def _association(NAMsvd, M, r, y, batches, ks=None, Nnull=1000, local_test=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = len(y)
    if ks is None:
        incr = max(int(0.02*n), 1)
        maxnpcs = min(4*incr, int(n/5))
        ks = np.arange(incr, maxnpcs+1, incr)

    # prep data
    (U, sv, V) = NAMsvd
    y = (y - y.mean())/y.std()

    def reg(q, k):
        Xpc = U[:,:k]
        gamma = Xpc.T.dot(q)
        qhat = Xpc.dot(gamma)
        return qhat, gamma

    def ftest(yhat, ycond, k):
        ssefull = (yhat - ycond).dot(yhat - ycond)
        ssered = ycond.dot(ycond)
        deltasse =  ssered - ssefull
        f = (deltasse / k) / (ssefull/n)
        return st.f.sf(f, k, n-(1+r+k))

    def minp_f(z):
        zcond = M.dot(z)
        zcond = zcond / zcond.std()
        ps = np.array([ftest(reg(zcond, k)[0], zcond, k) for k in ks])
        return ks[np.argmin(ps)], ps[np.argmin(ps)], ps
    # TODO: return correlation btwn ycond and yhat

    # get non-null f-test p-value
    k, p, ps, = minp_f(y)

    # compute final p-value using Nnull null f-test p-values
    y_ = stats.conditional_permutation(batches, y, Nnull)
    nullminps = np.array([minp_f(y__)[1] for y__ in y_.T])
    pfinal = ((nullminps <= p+1e-8).sum() + 1)/(Nnull + 1)

    # get neighborhood scores
    ycond = M.dot(y)
    ycond /= ycond.std()
    yhat, gamma = reg(ycond, k)
    ncorrs = (np.sqrt(sv[:k])*gamma/n).dot(V[:,:k].T)

    # get neighborhood fdrs if requested
    fdrs, fdr_5p_t, fdr_10p_t = None, None, None
    if local_test:
        print('computing neighborhood-level FDRs')
        Nnull = min(1000, Nnull)
        y_ = y_[:,:Nnull]
        ycond_ = M.dot(y_)
        ycond_ /= ycond_.std(axis=0)
        gamma_ = U[:,:k].T.dot(ycond_) / len(ycond_)
        nullncorrs = np.abs(V[:,:k].dot(np.sqrt(sv[:k])[:,None]*gamma_))

        fdr_thresholds = np.arange(np.abs(ncorrs).max()/4, np.abs(ncorrs).max(), 0.005)
        fdr_vals = stats.empirical_fdrs(ncorrs, nullncorrs, fdr_thresholds)

        fdrs = pd.DataFrame({
            'threshold':fdr_thresholds,
            'fdr':fdr_vals,
            'num_detected': [(np.abs(ncorrs)>t).sum() for t in fdr_thresholds]})

        # find maximal FDR<5% and FDR<10% sets
        if np.min(fdrs.fdr)>0.05:
            fdr_5p_t = None
        else:
            fdr_5p_t = fdrs[fdrs.fdr <= 0.05].iloc[0].threshold
        if np.min(fdrs.fdr)>0.1:
            fdr_10p_t = None
        else:
            fdr_10p_t = fdrs[fdrs.fdr <= 0.1].iloc[0].threshold

        del gamma_, nullncorrs

    del y_

    res = {'p':pfinal, 'nullminps':nullminps, 'k':k, 'ncorrs':ncorrs, 'fdrs':fdrs,
            'fdr_5p_t':fdr_5p_t, 'fdr_10p_t':fdr_10p_t,
			'yresid_hat':yhat, 'yresid':ycond, 'ks':ks}
    return Namespace(**res)

def association(data, y, batches, covs, nam_nsteps=None, max_frac_pcs=0.15, suffix='',
    force_recompute=False, **kwargs):
    du = data.uns
    npcs = np.max([10, int(max_frac_pcs * len(y))]) #Modified 
    if force_recompute or \
        'NAMqc'+suffix not in du or \
        not np.allclose(batches, du['batches'+suffix]):
        print('qcd NAM not found; computing and saving')
        NAM = nam(data, nsteps=nam_nsteps)
        NAMqc, keep = _qc(NAM, batches)
        du['NAMqc'+suffix] = NAMqc
        du['keptcells'+suffix] = keep
        du['batches'+suffix] = batches

    def samecovs(A, B):
        if A is None: A = np.zeros(0)
        if A.shape == B.shape:
            return np.allclose(A, B)
        else:
            return False
    if force_recompute or \
        'NAMsvdU'+suffix not in du or \
        not samecovs(covs, du['covs'+suffix]):
        print('covariate-adjusted NAM not found; computing and saving')
        NAMsvd, M, r = _prep(du['NAMqc'+suffix], covs, batches)
        du['NAMsvdU'+suffix] = NAMsvd[0]
        du['NAMsvdsvs'+suffix] = NAMsvd[1]
        du['NAMsvdV'+suffix] = NAMsvd[2][:,:npcs]
        du['M'+suffix] = M
        du['r'+suffix] = r
        du['covs'+suffix] = (np.zeros(0) if covs is None else covs)

    # do association test
    NAMsvd = (du['NAMsvdU'+suffix], du['NAMsvdsvs'+suffix], du['NAMsvdV'+suffix])
    res = _association(NAMsvd, du['M'+suffix], du['r'+suffix],
                        y, batches, **kwargs)

    # add info about kept cells
    vars(res)['kept'] = du['keptcells'+suffix]

    return res
