def analyze(
    a, # adjacency matrix, cells x cells
    Y, # outcome, samples x 1 TODO: allow for cell-level outcome
    C=None, # sample sizes, samples x 1
    B=None, # batch ids, samples x 1
    T=None, # sample-level covariates, samples x num_sample_covariates
    s=None): # cell-level covariates, cells x num_cell_covariates
    pass

def make_graph(sc):
    pass
