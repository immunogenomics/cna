"""Regression tests for the pandas>=2.0 compatibility fix in cna.tools._association.

Background: _resid_nam() returns the residualising projection M as a pandas
DataFrame, so inside _association() every quantity derived from it (ycond,
zcond, ycond_) used to be a Series/DataFrame. On pandas >= 2.0 the line
`ncorrs = (ycond[:,None]*NAMresid).mean(axis=0)` raises ValueError, because
indexing a Series with a tuple was removed in pandas 2.0.

The fix converts M to an ndarray once, and pins ddof=1 on the standard
deviations of the derived quantities so that the numbers are unchanged from
the pandas 1.x behaviour (numpy's std defaults to ddof=0, pandas' to ddof=1;
without the pin, ncorrs/coef would be rescaled by sqrt(n/(n-1))).
"""
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import pytest

import cna


def make_data(n_cells=6000, n_samples=24, n_pcs=10, seed=0):
    rng = np.random.default_rng(seed)
    sid = np.repeat([f's{i}' for i in range(n_samples)], n_cells // n_samples)
    X = rng.normal(size=(len(sid), n_pcs))
    # a real signal: half the samples get a shifted cluster
    case = np.isin(sid, [f's{i}' for i in range(n_samples // 2)])
    X[case & (rng.random(len(sid)) < 0.3), 0] += 3
    a = ad.AnnData(
        obs=pd.DataFrame(
            {'sid': sid, 'donor': np.char.replace(sid, 's', 'd')},
            index=[str(i) for i in range(len(sid))]),
        obsm={'X_pca': X})
    sc.pp.neighbors(a, n_neighbors=15, use_rep='X_pca', random_state=0)
    samples = pd.Index(sorted(set(sid)))
    y = pd.Series((samples.str.slice(1).astype(int) < n_samples // 2).astype(float),
                  index=samples)
    return a, y, samples


def kwargs_for(mode, y, samples):
    """Returns (y, kwargs) for each of the four call paths."""
    rng = np.random.default_rng(1)
    kw = {}
    if mode == 'batches':
        kw['batches'] = pd.Series(rng.integers(0, 2, len(samples)).astype(float),
                                  index=samples)
    if mode == 'covs':
        kw['covs'] = pd.DataFrame({'x': rng.normal(size=len(samples))}, index=samples)
    if mode == 'donorids':
        # two samples per donor, y constant within donor
        donor = pd.Series([f'd{i//2}' for i in range(len(samples))], index=samples)
        y = donor.map(lambda d: float(int(d[1:]) % 2))
        kw['donorids'] = donor
    return y, kw


@pytest.mark.parametrize('mode', ['plain', 'batches', 'covs', 'donorids'])
def test_association_runs_on_pandas2(mode):
    a, y, samples = make_data()
    y, kw = kwargs_for(mode, y, samples)
    res = cna.tl.association(a, y, 'sid', return_full=True, Nnull=200, seed=0, **kw)

    assert 0 < res.p <= 1
    assert isinstance(res.ncorrs, pd.Series) and len(res.ncorrs) == res.kept.sum()
    assert np.isfinite(res.ncorrs.values).all()
    # ncorrs must stay indexed by neighbourhood, since association() assigns it
    # into data.obs by label
    assert res.ncorrs.index.equals(res.namresid.columns)
    assert 'coef' in a.obs and 'coef_fdr' in a.obs
    assert a.obs.loc[res.kept, 'coef'].notna().all()
    # NB: empirical_fdrs() returns an unclipped mean FDP, which legitimately
    # exceeds 1 for a null trait, so only non-negativity/finiteness is asserted.
    assert (a.obs['coef_fdr'] >= 0).all()
    assert np.isfinite(a.obs['coef_fdr']).all()


def test_association_detects_planted_signal():
    a, y, _ = make_data()
    res = cna.tl.association(a, y, 'sid', Nnull=500, seed=0, return_full=True)
    assert res.p < 0.05      # planted shift in half the samples must be found
    assert res.fdrs.fdr.min() < 0.05          # and neighbourhoods pass local FDR
    assert (a.obs['coef_fdr'] < 0.05).sum() > 0


def test_association_null_is_calibrated():
    a, y, samples = make_data()
    rng = np.random.default_rng(2)
    ps = [cna.tl.association(a, pd.Series(rng.normal(size=len(samples)), index=samples),
                             'sid', Nnull=200, seed=0)
          for _ in range(10)]
    # random traits are not all significant
    assert min(ps) > 1 / 201 and np.mean(ps) > 0.2


def test_residualized_y_uses_ddof1():
    """Locks in the pandas-1.x normalisation of ycond.

    Guards against a 'fix' that converts M to ndarray without pinning ddof,
    which would rescale yresid/ncorrs/beta/FDR thresholds by sqrt(n/(n-1)).
    """
    a, y, _ = make_data()
    res = cna.tl.association(a, y, 'sid', return_full=True, Nnull=100, seed=0)
    n = len(res.yresid)
    assert np.isclose(np.std(res.yresid, ddof=1), 1.0, atol=1e-12)
    assert not np.isclose(np.std(res.yresid, ddof=0), 1.0, atol=1e-6)
    assert np.isclose(np.std(res.yresid, ddof=0), np.sqrt((n - 1) / n), atol=1e-12)


def test_ncorrs_matches_definition():
    """ncorrs is the mean over samples of yresid * residualized NAM."""
    a, y, _ = make_data()
    res = cna.tl.association(a, y, 'sid', return_full=True, Nnull=100, seed=0)
    expected = (np.asarray(res.yresid)[:, None] * res.namresid).mean(axis=0)
    assert np.allclose(res.ncorrs.values, expected.values, rtol=0, atol=1e-12)


def test_association_accepts_ndarray_M():
    """_association must give identical results whether M is a DataFrame or ndarray."""
    from cna.tools._nam import nam, _resid_nam
    from cna.tools._association import check_inputs, reindex_and_filter_nam
    from cna.tools._association import _association

    a, y, _ = make_data()
    batches, filter_samples = check_inputs(a, y, 'sid', None, None, None, False)
    NAM, kept = nam(a, 'sid', batches=batches)
    NAM, kept, batches, covs, donorids, filter_samples = reindex_and_filter_nam(
        NAM, kept, y, batches, None, None, filter_samples)
    N = filter_samples.sum()
    r = _resid_nam(NAM, None, None, npcs=min(N, max(10, int(0.15 * N))))

    svd = (r.namresid_sampleXpc.values, r.namresid_svs.values, r.namresid_nbhdXpc.values)
    common = dict(NAMsvd=svd, NAMresid=r.namresid, r=r.r,
                  y=y[filter_samples].values, batches=batches[filter_samples].values,
                  donorids=None, Nnull=200, seed=0)

    assert isinstance(r.M, pd.DataFrame)          # _resid_nam still returns a labelled M
    a_df = _association(M=r.M, **common)
    a_np = _association(M=np.asarray(r.M), **common)

    assert a_df.p == a_np.p and a_df.k == a_np.k
    assert np.allclose(a_df.ncorrs.values, a_np.ncorrs.values, rtol=0, atol=1e-12)


def test_nam_is_float_and_finite():
    a, y, _ = make_data()
    NAM, kept = cna.tl.nam(a, 'sid')
    assert NAM.dtypes.eq(float).all()
    assert np.isfinite(NAM.values).all()
