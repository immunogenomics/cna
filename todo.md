SIMULATIONS
Datasets
- Synthetic single-cell: not sure if there's a role
- TBRU
    * Semi-synthetic single-cell, synthetic phenotypes (Nghia)
    * Real single-cell (TBRU), synthetic phenotypes (probably preferable)
- Down-sampled TBRU to assess power only

Scenarios
- {null, null w/confounding}
- {differential abundance, cell-type-specific differential expression}x{one of Aparna’s clusters, union of two clusters, subset of a cluster}
- {shift along innateness gradient}

Methods
- ours
- MASC w/expert (i.e., Aparna) clusters
- MASC with clustering using default
  parameters

METHODS QUESTIONS - OPEN
- is it worth thinking about creating an effect size in addition to a z-score? For, e.g., comparison of nested models.
- does pre- vs post-harmony make a difference?
- what should we do with the cell-level p-values? UMAP them? dPCA them? differential abundance?
- does number of neighbors in umap graph matter?
- does the strength of self loops matter? (guess: not really)

METHODS QUESTIONS - TENTATIVELY RESOLVED
- how many steps should the diffusion take? exponential growth of results (might later consider surface area/volume of sig cells or timpoint with lowest single estimated FWER across cells)
- how many null permutations can we get down to while still having accurate p-values? (100 for now, needs confirmatory analysis)
- how should we assess significance? FWER for now (leaving FDR and P(FDP < 5%) implemented)

DOWNSTREAM METHODS PROJECTS
- Input different features (e.g., dPCs, pairwise products) to diffusion to find different kinds of signal
- computational efficiency: first cluster any set of cells into, say, 10K tiny clusters?

DATASETS
- TBRU x 3 modalities
    - TB, sex, age, ancestry
- MASC data set: RA case/ctrl
- Abundance QTLs?
- Cell-type-specific eQTLs a la Kaz? maybe later
- Ebola data set? talk to dylan
- V2F iPSC data set? (single cell + cell morphology via cellpainter; maybe later)
- (Non-single-cell data sets?)

