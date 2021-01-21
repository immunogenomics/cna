import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def umap_ncorr(data, res, fdr_thresh=None, **kwargs):
    if fdr_thresh is None:
        fdr_thresh = 0.05

    passed = res.fdrs[res.fdrs.fdr <= fdr_thresh]
    if len(passed) == 0:
        print('no neighborhoods were significant at FDR <', fdr_thresh)
        return
    else:
        thresh = passed.threshold.iloc[0]

    ix1 = np.repeat(False, len(data))
    ix1[res.kept] = np.abs(res.ncorrs) > thresh
    c = res.ncorrs[np.abs(res.ncorrs) > thresh]

    umap_overlay(data, ix1, c, **kwargs)

def umap_overlay(data, ix1, c,
    scatter0={},
    scatter1={},
    ax=None, noframe=True, colorbar=True, **cbar_kw):

    # set default plotting options
    scatter0_ = {'alpha':0.1, 's':2, 'color':'grey'}
    scatter1_ = {'alpha':0.2, 's':8, 'c':c, 'cmap':'seismic',
                    'vmin':-np.abs(c).max(),
                    'vmax':np.abs(c).max()}
    scatter0_.update(scatter0)
    scatter1_.update(scatter1)

    if ax is None:
        ax = plt.gca()

    # do plotting
    ax.scatter(*data.obsm['X_umap'].T, **scatter0_)
    res = ax.scatter(*data.obsm['X_umap'][ix1].T, **scatter1_)

    # remove ticks and spines
    ax.set_xticks([]); ax.set_yticks([])
    if noframe:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # add colorbar if needed
    if colorbar:
        fig = plt.gcf()
        cbar = fig.colorbar(res, ax=ax, **cbar_kw)
        cbar.set_alpha(1)
        cbar.draw_all()
    else:
        cbar = None

    return ax, cbar
