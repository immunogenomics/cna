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

METHODS QUESTIONS
- does using dPCs first improve results substantially?
- how many steps should the diffusion take? (use z-score correlation)
- how many null permutations can we get down to while still having accurate
  p-values? (maybe 100?)
- how should we assess significance? FDR, naive correlated Bonferroni, other
  Bonferronis?
- does pre- vs post-harmony make a difference?
- does number of neighbors in umap graph matter?
- does the strength of self loops matter? (guess: not really)
- computational efficiency: first cluster any set of cells into, say, 10K tiny clusters?

DATASETS
- TBRU
    - TB, sex, age, ancestry
- MASC data set: RA case/ctrl
- Abundance QTLs?
- Cell-type-specific eQTLs a la Kaz? not yet
- Ebola data set? talk to dylan
- (Non-single-cell data sets?)

