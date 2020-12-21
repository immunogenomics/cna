import numpy as np

# compute sample size for each sample
def sample_sizes(data, sampleid='id'):
    return data.obs[sampleid].value_counts()
