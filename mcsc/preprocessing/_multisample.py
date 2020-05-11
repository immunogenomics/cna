import numpy as np

def sortedcopy(data, uns_samples='sampleXmeta', samplename='id'):
    """
    data is an AnnData object that must have: 1) an uns field that stores
        sample-level data, 2) obs columns corresponding to sample id and batch id.
    warning: this method returns a copy of the data set.
    """
    ix = np.argsort(data.obs[samplename].values.astype(str))
    data = data[ix].copy()
    # anndata takes care of reordering things in obsm but not in uns
    if 'neighbors' in data.uns:
        for k in ['connectivities', 'distances']:
            data.uns['neighbors'][k] = data.uns['neighbors'][k][ix][:,ix]

    if uns_samples in data.uns:
        ix = np.argsort(data.uns[uns_samples].index.values.astype(str))
        data.uns[uns_samples] = data.uns[uns_samples].iloc[ix]

    return data

def issorted(data, uns_samples='sampleXmeta', samplename='id', batchname='batch'):
    """
    data is an AnnData object that must have: 1) an uns field that stores
        sample-level data, 2) obs columns corresponding to sample id and batch id.
    """
    # check that cells are sorted by sample
    sampleids = data.obs[samplename].values
    if sum(sampleids[:-1] != sampleids[1:]) >= len(np.unique(sampleids)):
        print('ERROR: cells not sorted by sample')
        return False

    # check that sorting of samples in cell data matches that in sample data
    sampleXmeta = data.uns[uns_samples]
    a = data.obs[[samplename]].drop_duplicates().values
    b = sampleXmeta.reset_index()[[samplename]].values
    if len(a) != len(b):
        print('ERROR: different number of distinct samples in data.obs and sampleXmeta')
        return False
    elif not (data.obs[[samplename]].drop_duplicates().values == \
        sampleXmeta.reset_index()[[samplename]].values).all():
        print('ERROR: sample order of cells does not match sample order of samples')
        return False

    return True

def sample_size(data, uns_samples='sampleXmeta', samplename='id'):
    # compute sample size for each sample
    sampleids = data.obs[samplename].values
    samplechanges = np.concatenate([
            [0],
            np.where(sampleids[:-1] != sampleids[1:])[0] + 1,
            [len(data)]
        ])
    return np.diff(samplechanges)
