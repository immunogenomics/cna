import numpy as np

# converts per-cell metadata into per-sample metadata using the function specified in
#  aggregate. columns can be either a string or an iterable of strings
def cell_to_sample(data, columns, aggregate=np.mean,
        sampleXmeta='sampleXmeta', sampleid='id'):
    if type(columns) == str:
        columns = [columns]
    for c in columns:
        data.uns[sampleXmeta][c] = \
            data.obs[[sampleid, c]].groupby(by=sampleid).aggregate(aggregate)

# compute sample size for each sample
def sample_sizes(data, sampleid='id'):
    return data.obs[sampleid].value_counts()
