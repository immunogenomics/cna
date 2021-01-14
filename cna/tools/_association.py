import numpy as np
import pandas as pd
import scipy.stats as st
import gc, warnings
from argparse import Namespace
import cna.tools._stats as stats
from ._nam import nam, _df_to_array

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
    notnan = ~np.isnan(y)
    y = y[notnan]; batches = batches[notnan]
    U = U[notnan]; M = M[notnan][:,notnan] #TODO: think a bit more about how to interpret
    y = (y - y.mean())/y.std()

    def _reg(q, k):
        Xpc = U[:,:k]
        beta = Xpc.T.dot(q)
        qhat = Xpc.dot(beta)
        return qhat, beta

    def _r2(q, k):
        qhat, _ = _reg(q, k)
        return ((q - q.mean()).dot(qhat - qhat.mean()) / q.std() / qhat.std() / len(q))**2

    def _ftest(yhat, ycond, k):
        ssefull = (yhat - ycond).dot(yhat - ycond)
        ssered = ycond.dot(ycond)
        deltasse =  ssered - ssefull
        f = (deltasse / k) / (ssefull/n)
        p = st.f.sf(f, k, n-(1+r+k))
        return p

    def _minp_f(z):
        zcond = M.dot(z)
        zcond = zcond / zcond.std()
        ps = np.array([
            _ftest(
                _reg(zcond, k)[0],
                zcond,
                k)
            for k in ks
        ])
        return ks[np.argmin(ps)], ps[np.argmin(ps)], ps

    # get non-null f-test p-value
    k, p, ps, = _minp_f(y)

    # compute coefficients and r2 with chosen model
    ycond = M.dot(y)
    ycond /= ycond.std()
    yhat, beta = _reg(ycond, k)
    r2_perpc = (beta / np.sqrt(len(ycond)))**2
    r2 = _r2(ycond, k)

    # get neighborhood scores with chosen model
    ncorrs = (np.sqrt(sv[:k])*beta/n).dot(V[:,:k].T)

    # compute final p-value using Nnull null f-test p-values
    y_ = stats.conditional_permutation(batches, y, Nnull)
    nullminps = np.array([_minp_f(y__)[1] for y__ in y_.T])
    pfinal = ((nullminps <= p+1e-8).sum() + 1)/(Nnull + 1)
    if (nullminps <= p+1e-8).sum() == 0:
        warnings.warn('global association p-value attained minimal possible value. '+\
                'Consider increasing Nnull')

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
			'yresid_hat':yhat, 'yresid':ycond, 'ks':ks, 'beta':beta,
            'r2':r2, 'r2_perpc':r2_perpc}
    return Namespace(**res)

def association(data, y, batches=None, covs=None, nsteps=None, suffix='',
    force_recompute=False, **kwargs):

    # formatting and error checking
    if batches is None:
        batches = np.ones(len(data.samplem))
    covs = _df_to_array(data, covs)
    batches = _df_to_array(data, batches)
    y = _df_to_array(data, y)

    du = data.uns
    nam(data, batches=batches, covs=covs, nsteps=nsteps, suffix=suffix,
                    force_recompute=force_recompute)
    NAMsvd = (
        du['NAM_sampleXpc'+suffix].values,
        du['NAM_svs'+suffix],
        du['NAM_nbhdXpc'+suffix].values
        )
    res = _association(NAMsvd, du['_M'+suffix], du['_r'+suffix], y, batches, **kwargs)

    # add info about kept cells
    vars(res)['kept'] = du['keptcells'+suffix]

    return res
