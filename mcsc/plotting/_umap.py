import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def umap(data, ix, c,
    scatter0={},
    scatter1={},
    ax=None, noframe=True, colorbar=True, **cbar_kw):
    # set default plotting options
    scatter0_ = {'alpha':0.01, 's':2, 'color':'grey'}
    scatter1_ = {'alpha':0.1, 's':4, 'c':c[ix], 'cmap':'seismic',
                    'vmin':-np.abs(c[ix]).max(),
                    'vmax':np.abs(c[ix]).max()}
    scatter0_.update(scatter0)
    scatter1_.update(scatter1)

    if ax is None:
        ax = plt.gca()

    # do plotting
    ax.scatter(*data.obsm['X_umap'].T, **scatter0_)
    res = ax.scatter(*data.obsm['X_umap'][ix].T, **scatter1_)

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

def zhists(z, ts, overlays=[], propsig=None, **kwargs):
    df = pd.DataFrame(np.concatenate([z[t] for t in ts]), columns=['z'])
    df['t'] = np.concatenate([[t]*z.shape[1] for t in ts])
    plt.figure(figsize=(20,10))
    sns.violinplot(x='t', y='z', data=df, **kwargs)
    for o in overlays:
        plt.plot(o[ts], color='k')
    ax1 = plt.gca()
    ax2 = None

    if propsig is not None:
        ax2 = plt.gca().twinx()
        ax2.plot(propsig[ts], color='blue')
        ax2.set_ylim(-1.1,1.1)
        ax2.tick_params(axis='y', labelcolor='blue')

    return ax1, ax2

