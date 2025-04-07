import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

def umap_ncorr(data, fdr_thresh=None, key='coef', **kwargs):
    if fdr_thresh is None:
        fdr_thresh = 0.1

    passed = data.obs[f'{key}_fdr'] <= fdr_thresh
    if len(passed) == 0:
        print('no neighborhoods were significant at FDR <', fdr_thresh)

    umap_overlay(data, passed, key, **kwargs)

def umap_overlay(data, mask, key,
    scatter0={},
    scatter1={},
    ax=None, noframe=True):

    # set default plotting options
    if ax is None:
        ax = plt.gca()
    c = data.obs[mask][key]
    scatter0_ = {'alpha':0.8, 's':2}
    scatter1_ = {'alpha':0.9, 's':8, 'cmap':'seismic',
                    'vmin':-np.abs(c).max() if len(c) > 0 else 0,
                    'vmax':np.abs(c).max() if len(c) > 0 else 1}
    scatter0_.update(scatter0)
    scatter1_.update(scatter1)

    # do plotting
    sc.pl.umap(data, ax=ax, show=False, **scatter0_)
    sc.pl.umap(data[mask], color=key, ax=ax, show=False, title='', **scatter1_)

    return ax
