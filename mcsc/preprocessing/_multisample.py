import numpy as np

def issorted(data, uns_samples='sampleXmeta', samplename='id', batchname='batch'):
    """
    data is an AnnData object that must have: 1) an uns field that stores
        sample-level data, 2) obs columns corresponding to sample id and batch id.
    """
    # check that cells are sorted by sample and then by batch
    sampleids = data.obs[samplename].values
    batchids = data.obs[batchname].values
    if sum(sampleids[:-1] != sampleids[1:]) >= len(np.unique(sampleids)) or \
        sum(batchids[:-1] > batchids[1:]) >= len(np.unique(batchids)):
        return False

    # check that sorting of samples/batches in cell data matches that in sample data
    sampleXmeta = data.uns[uns_samples]
    a = data.obs[[samplename, batchname]].drop_duplicates().values
    b = sampleXmeta.reset_index()[[samplename, batchname]].values
    if len(a) != len(b):
        return False
    elif not (data.obs[[samplename, batchname]].drop_duplicates().values == \
        sampleXmeta.reset_index()[[samplename, batchname]].values).all():
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
