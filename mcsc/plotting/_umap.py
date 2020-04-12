import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def umap(data, ix, c, alpha0=0.01, alpha1=0.1, ax=None, colorbar=True):
    if ax is None:
        ax = plt.gca()
    ax.scatter(*data.obsm['X_umap'].T, color='grey', s=3, alpha=alpha0)
    ax.scatter(*data.obsm['X_umap'][ix].T, c=c[ix],
        vmin=-np.abs(c[ix]).max(), vmax=np.abs(c[ix]).max(),
        s=4, alpha=alpha1, cmap='seismic')
    if colorbar:
        fig = plt.gcf()
        fig.colorbar(ax=ax)

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

