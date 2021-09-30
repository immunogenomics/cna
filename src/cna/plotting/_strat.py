import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# build violin plot that stratifies neighborhood coefficients by a discretization of the data
#   such as a clustering. Any additional keyword arguments are passed to
#   matplotlib.violinplot
def violinplot(data, res, stratification, ax=None, cmap='seismic', **kwargs):
    if ax is None:
        ax = plt.gca()
    kwargs_ = {
            'widths': 0.9,
            'showmeans': False,
            'showextrema': False,
            'showmedians': False,
        }
    kwargs_.update(kwargs)

    bins = data.obs[stratification].unique()
    violins = ax.violinplot([res.ncorrs[data.obs[stratification]==v] for v in bins],
                            np.arange(len(bins)),
                            **kwargs_)
    (ymin, ymax), (xmin, xmax) = ax.get_ylim(), ax.get_xlim()
    Nx, Ny = 1,1000
    imgArr = np.tile(np.linspace(0, 1, Ny), (Nx, 1)).T

    for violin in violins['bodies']:
        path = Path(violin.get_paths()[0].vertices)
        patch = PathPatch(path, facecolor='none', edgecolor='none')
        ax.add_patch(patch)
        img = ax.imshow(imgArr,
                        origin='lower',
                        extent=[xmin,xmax,ymin,ymax],
                        aspect='auto',
                        cmap=cmap,
                        clip_path=patch)
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels(bins)
    ax.set_xlabel(stratification)
    ax.set_ylabel('Neighborhood Coefficient')

    return ax
