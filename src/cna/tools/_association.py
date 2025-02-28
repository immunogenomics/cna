import numpy as np
import pandas as pd
import scipy.stats as st
import gc, warnings
from argparse import Namespace
from ._stats import conditional_permutation, grouplevel_permutation, empirical_fdrs
from ._nam import nam, _df_to_array
from ._out import select_output

def _association(NAMsvd, NAMresid, M, r, y, batches, donorids, ks=None, Nnull=1000, force_permute_all=False,
                    local_test=True, seed=None, show_progress=False):
    # output level
    out = select_output(show_progress)
    
    if seed is not None:
        np.random.seed(seed)
    if force_permute_all:
        batches = np.ones(len(y))

    # prep data
    (U, sv, V) = NAMsvd
    y = (y - y.mean())/y.std()
    n = len(y)

    if ks is None:
        incr = max(int(0.02*n), 1)
        maxnpcs = min(4*incr, int(n/5))
        ks = np.arange(incr, maxnpcs+1, incr)
    if max(ks) + r >= n:
        raise ValueError(
                    'Maximum number of PCs plus number of covariates must be less than n-1. '+\
                    f'Currently it is {max(ks)+r} while n is {n}. Either reduce the number of covariates '+\
                    'or reduce the number of PCs to consider using the optional argument ks=[...].')

    def _reg(q, k):
        Xpc = U[:,:k]
        beta = Xpc.T.dot(q) #Xpc.T.dot(Xpc) = I so no need to compute it
        qhat = Xpc.dot(beta)
        return qhat, beta

    def _stats(yhat, ycond, k):
        ssefull = (yhat - ycond).dot(yhat - ycond)
        ssered = ycond.dot(ycond)
        deltasse =  ssered - ssefull
        f = (deltasse / k) / (ssefull/n)
        p = st.f.sf(f, k, n-(1+r+k)) # F test
        r2 = 1 - ssefull/ssered
        return p, r2

    def _minp_stats(z):
        zcond = M.dot(z)
        zcond = zcond / zcond.std()
        ps, r2s = np.array([
            _stats(
                _reg(zcond, k)[0],
                zcond,
                k)
            for k in ks
        ]).T
        k_ = np.nanargmin(ps)
        return ks[k_], ps[k_], r2s[k_]

    # get non-null f-test p-value
    k, p, r2 = _minp_stats(y)
    if k == max(ks):
        warnings.warn(('data supported use of {} NAM PCs, which is the maximum considered. '+\
            'Consider allowing more PCs by using the "ks" argument.').format(k))

    # compute coefficients and r2 with chosen model
    ycond = M.dot(y)
    ycond /= ycond.std()
    yhat, beta = _reg(ycond, k)
    _, fullbeta = _reg(ycond, n)
    r2_perpc = (beta / np.sqrt(ycond.dot(ycond)))**2

    # get neighborhood coefficients
    ncorrs = (y[:,None]*NAMresid).mean(axis=0)

    # compute final p-value using Nnull null f-test p-values
    if donorids is not None:
        y_ = grouplevel_permutation(donorids, y, Nnull)
    else:
        y_ = conditional_permutation(batches, y, Nnull)
    nullminps, nullr2s = np.array([_minp_stats(y__)[1:] for y__ in y_.T]).T
    pfinal = ((nullminps <= p+1e-8).sum() + 1)/(Nnull + 1)
    if (nullminps <= p+1e-8).sum() == 0:
        warnings.warn('global association p-value attained minimal possible value. '+\
                'Consider increasing Nnull')

    # get neighborhood fdrs if requested
    fdrs, fdr_5p_t, fdr_10p_t = None, None, None
    if local_test:
        print('computing neighborhood-level FDRs', file=out)
        Nnull = min(1000, Nnull)
        y_ = y_[:,:Nnull]
        ycond_ = M.dot(y_)
        ycond_ /= ycond_.std(axis=0)
        gamma_ = U[:,:k].T.dot(ycond_)
        nullncorrs = np.abs(NAMresid.T.dot(ycond_) / n)

        maxcorr = max(np.abs(ncorrs).max(), 0.001)
        fdr_thresholds = np.arange(maxcorr/4, maxcorr, maxcorr/400)
        fdr_vals = empirical_fdrs(ncorrs, nullncorrs, fdr_thresholds)

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
            'r2':r2, 'r2_perpc':r2_perpc,
            'nullr2_mean':nullr2s.mean(), 'nullr2_std':nullr2s.std()}
    return Namespace(**res)

def association(data, y, batches=None, covs=None, donorids=None, nsteps=None, suffix='',
    force_recompute=False, show_progress=False, allow_low_sample_size=False, **kwargs):
    # output level
    out = select_output(show_progress)
    
    # formatting and error checking
    if batches is not None and donorids is not None:
        raise ValueError('CNA does not currently support conditioning on batch '+\
            'while also accounting for multiple samples per donor')

    if batches is None:
        batches = np.ones(data.N)
    covs = _df_to_array(data, covs)
    batches = _df_to_array(data, batches)
    y = _df_to_array(data, y)
    if y.shape != (data.N,):
        raise ValueError(
            'y should be an array of length data.N; instead its shape is: '+str(y.shape))
    if data.N < 10 and not allow_low_sample_size:
        raise ValueError(
            'Dataset has fewer than 10 samples. CNA may have poor power at low sample sizes '+\
            'because its null distribution is one in which each sample\'s single-cell profile '+\
            'is unchanged but the sample labels are randomly assigned. If you want to run CNA '+\
            'at this sample size despite the possibility of low power, you can do so by '+\
            'invoking the association(...) function with the argument '+\
            'allow_low_sample_size=True.')

    if covs is not None:
        filter_samples = ~(np.isnan(y) | np.any(np.isnan(covs), axis=1))
        if donorids is not None:
            print('WARNING: CNA currently does not account for multiple samples per donor '+\
                'when conditioning on covariates. This conditioning may therefore account '+\
                'only incompletely for the covariates of interest. We expect this to make '+\
                'only minor differences in most cases, but we have not investigated it formally')
    else:
        filter_samples = ~np.isnan(y)

    du = data.uns
    nam(data, batches=batches, covs=covs, filter_samples=filter_samples,
                    nsteps=nsteps, suffix=suffix, force_recompute=force_recompute,
                    show_progress=show_progress,
                    **kwargs)
    NAMsvd = (
        du['NAM_sampleXpc'+suffix].values,
        du['NAM_svs'+suffix],
        du['NAM_nbhdXpc'+suffix].values
        )

    print('performing association test', file=out)
    res = _association(NAMsvd, du['NAM_resid.T'+suffix].T, du['_M'+suffix], du['_r'+suffix],
        y[du['_filter_samples'+suffix]], batches[du['_filter_samples'+suffix]],
        donorids[du['_filter_samples'+suffix]] if donorids is not None else None,
        show_progress=show_progress,
        **kwargs)

    # add info about kept cells
    vars(res)['kept'] = du['keptcells'+suffix]

    return res
