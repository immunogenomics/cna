import numpy as np

# copies sample-level metadata that's stored per-cell from d.obs to sampleXmeta
def cell_to_sample(data, columns, sampleXmeta='sampleXmeta', sampleid='id'):
    if type(columns) == str:
        columns = [columns]
    for c in columns:
        data.uns[sampleXmeta][c] = \
            data.obs[[sampleid, c]].drop_duplicates().set_index(sampleid)

# compute sample size for each sample
def sample_sizes(data, sampleid='id'):
    return data.obs[sampleid].value_counts()
