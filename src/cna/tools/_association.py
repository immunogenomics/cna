import numpy as np
import pandas as pd
import scipy.stats as st
import gc, warnings
from argparse import Namespace
from ._stats import conditional_permutation, grouplevel_permutation, empirical_fdrs
from ._nam import nam, _resid_nam
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
        maxnpcs = max(min(4*incr, int(n/5)), 1)
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
        fdr_vals = empirical_fdrs(ncorrs.values, nullncorrs.values, fdr_thresholds)

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

def association(data, y, sid_name, batches=None, covs=None, donorids=None, ks=None, key_added='coef',
                max_frac_pcs=0.15, nsteps=None, show_progress=False, allow_low_sample_size=False,
                return_full=False, ridges=None, **kwargs):
    out = select_output(show_progress)

    # Check formats of iputs
    if not isinstance(y, pd.Series):
        raise TypeError(f"'y' must be a pandas Series, but got {type(y)}")
    if batches is not None and not isinstance(batches, pd.Series):
        raise TypeError(f"'batches' must be a pandas Series, but got {type(batches)}")
    if covs is not None and not isinstance(covs, pd.DataFrame):
        raise TypeError(f"'covs' must be a pandas DataFrame, but got {type(covs)}")
    if donorids is not None and not isinstance(donorids, pd.Series):
        raise TypeError(f"'donorids' must be a pandas Series, but got {type(donorids)}")
    if not set(y.index).issubset(set(data.obs[sid_name])):
        print("WARNING: index of 'y' contains values not present in 'data[sid_name]'. These samples will be ignored.")
    if not set(data.obs[sid_name]).issubset(set(y.index)):
        raise ValueError("'data[sid_name]' contains values not present in the index of 'y'.")
    
    # Ensure batches and donorids not simultaneously set
    if batches is not None and donorids is not None:
        raise ValueError('We do not currently support conditioning on batch '+\
            'while also accounting for multiple samples per donor')

    if batches is None:
        batches = pd.Series(np.ones(len(y)), index=y.index)
    N = (~np.isnan(y)).sum()
    if N < 10 and not allow_low_sample_size:
        raise ValueError(
            'You are supplying phenotype information on fewer than 10 samples. This may lead to '+\
            'poor power at low sample sizes because its null distribution is one in which each '+\
            'sample\'s single-cell profile is unchanged but the sample labels are randomly '+\
            'assigned. If you want to run CNA at this sample size despite the possibility of low '+\
            'power, you can do so by invoking the association(...) function with the argument '+\
            'allow_low_sample_size=True.')

    if covs is not None:
        filter_samples = ~(y.isna() | covs.isna().any(axis=1)) & y.index.isin(data.obs[sid_name].unique())
        if donorids is not None:
            print('WARNING: CNA currently does not account for multiple samples per donor '+\
                'when conditioning on covariates. This conditioning may therefore account '+\
                'only incompletely for the covariates of interest. We expect this to make '+\
                'only minor differences in most cases, but we have not investigated it formally')
    else:
        filter_samples = ~np.isnan(y) & y.index.isin(data.obs[sid_name].unique())

    # compute NAM
    NAM, kept = nam(data, sid_name, batches=batches, nsteps=nsteps, show_progress=show_progress, **kwargs)
    
    # align all indices
    NAM = NAM.reindex(y.index)
    batches = batches.reindex(y.index)
    covs = covs.reindex(y.index) if covs is not None else None
    donorids = donorids.reindex(y.index) if donorids is not None else None
    filter_samples = filter_samples.reindex(y.index)

    # residualize NAM (after removing any columns that have zero variance after the sample filtering above)
    npcs = min(N, max([10]+[int(max_frac_pcs * N)]+[ks if ks is not None else []][0]))
    NAM = NAM[filter_samples]
    zero_variance_col_ix = np.where(NAM.std(axis=0) == 0)[0]
    kept[kept][zero_variance_col_ix] = False
    NAM = NAM.drop(columns=NAM.columns[zero_variance_col_ix])
    res = _resid_nam(NAM,
                            covs[filter_samples] if covs is not None else covs,
                            batches[filter_samples] if batches is not None else batches,
                            npcs=npcs,
                            ridges=ridges,
                            show_progress=show_progress)

    print('performing association test', file=out)
    res_ = _association(
        (res.namresid_sampleXpc.values, res.namresid_svs.values, res.namresid_nbhdXpc.values),
        res.namresid, res.M, res.r,
        y[filter_samples].values, batches[filter_samples].values,
        donorids[filter_samples].values if donorids is not None else None,
        show_progress=show_progress, ks=ks,
        **kwargs)
    res.__dict__.update(vars(res_)) # add info from from res_ to res
    res.nam = NAM
    res.kept = kept
    
    # store results at the neighborhood level
    if key_added in data.obs:
        warnings.warn(f"Key '{key_added}' already exists in data.obs. Overwriting.")
    data.obs[key_added] = np.NaN
    data.obs.loc[kept, key_added] = res.ncorrs
    
    # compute local FDRs
    def min_fdr_for_corr(ncorr):
        matching_fdrs = res.fdrs.loc[res.fdrs.threshold <= abs(ncorr)].fdr
        return matching_fdrs.min() if not matching_fdrs.empty else 1
    data.obs[f'{key_added}_fdr'] = data.obs[key_added].apply(min_fdr_for_corr)

    if return_full:
        return res
    else:
        return res.p