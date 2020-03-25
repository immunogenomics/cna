import matplotlib.pyplot as plt
import seaborn as sns

def umap(data, ix, c, thresh, alpha0=0.01, alpha1=0.1):
    plt.scatter(*data.obsm['X_umap'].T, color='grey', s=3, alpha=alpha0)
    plt.scatter(*data.obsm['X_umap'][ix].T, c=c[ix],
        vmin=-np.abs(c[ix]).max(), vmax=np.abs(c[ix]).max(),
        s=4, alpha=alpha1, cmap='seismic')
    plt.colorbar()
    plt.show()

def zhists(z, ts, overlays=[]):
    df = pd.DataFrame(np.concatenate([z[t] for t in ts]), columns=['z'])
    df['t'] = np.concatenate([[t]*z.shape[1] for t in ts])
    plt.figure(figsize=(20,10))
    sns.violinplot(x="t", y="z", data=df)
    for o in overlays:
        plt.plot(o[ix])
    plt.show()
