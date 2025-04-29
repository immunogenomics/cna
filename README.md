# cna
Covarying neighborhood analysis is a method for finding structure in- and conducting association analysis with multi-sample single-cell datasets. `cna` does not require a pre-specified transcriptional structure such as a clustering of the cells in the dataset. It aims instead to flexibly identify differences of all kinds between samples. `cna` is fast, does not require parameter tuning, produces measures of statistical significance for its association analyses, and allows for covariate correction.

`cna` is built on top of `scanpy` and offers a `scanpy`-like interface for ease of use.

If you prefer R, there is an [R implementation](https://github.com/korsunskylab/rcna) maintained separately by Ilya Korsunsky. (Though the R implementation may occasionally lag behind this implementation as updates are made.)

## installation
To use `cna`, you can either install it directly from the [Python Package Index](https://pypi.org/) by running, e.g.,

`pip install cna`

or if you'd like to manipulate the source code you can clone this repository and add it to your `PYTHONPATH`.

## demo
Take a look at our [tutorial](https://nbviewer.jupyter.org/github/yakirr/cna/blob/master/demo/demo.ipynb) to see how to get started with a small synthetic data set.

## talk
You can learn more about `cna` by watching our [talk](https://youtu.be/FlFYa79D4dc?t=2405) at the Broad Institute's Models, Inference, and Algorithms seminar, which is preceded by a [primer](https://youtu.be/FlFYa79D4dc) by Dylan Kotliar on nearest-neighbor graphs.

## notices
* April 29, 2025: We have made substantial changes to the `cna` API and are releasing a new package version 0.2.0. The main changes are that i) `cna` no longer will cache the NAM, and ii) `cna` will no longer use the MultiAnnData structure and will instead only use the standard scanpy AnnData structure. Code built for prior versions will likely not work for this new version, but should be easily adaptible by following the new demo.
* October 19, 2023: We have found a source of miscalibration in `cna`’s local association testing of individual neighborhoods that applies to unusual datasets, typically with limited sample size and very low complexity. This issue does not affect `cna`’s global test, which tests for aggregate association between single-cell profiles and a case-control phenotype; it only affects `cna`’s identification of which individual neighborhoods explain an aggregate association. The miscalibration appears mild on real datasets. However, in simulated datasets we observed miscalibration when i) the case-control phenotype was extremely highly correlated with the first principal component of the neighborhood abundance matrix, and ii) there were many neighborhoods lacking true associations to this phenotype. This issue has been fixed in `cna` version 0.1.6, which uses the full-rank rather than the rank-*k\** neighborhood abundance matrix to compute neighborhood coefficients. We re-ran the primary analyses from the index `cna` paper with this new version of `cna` and found that the results were broadly unchanged. Although `cna` found fewer FDR-significant neighborhoods in each dataset, it still found large numbers of neighborhoods corresponding to the key associated cell populations (albeit at FDR 10% rather than 5% for the dataset with the smallest sample size \[N=12\]). Additionally, the prior and updated neighborhood coefficients remain very similar (R>0.9 in all datasets). We did not modify CNA’s global test, which determines whether there is any association between the single cell profiles and the case-control phenotype, as that portion of the method is unaffected.
* January 20, 2022:  It has come to our attention that a bug introduced on July 16, 2021 caused `cna` to behave incorrectly for users with `anndata` version 0.7.2 or later, possibly resulting in false positive or false negative results. This bug was fixed in `cna` version 0.1.4. We strongly recommend that any users with `anndata` version 0.7.2 or later either re-clone `cna` or run `pip install --upgrade cna` and re-run all analyses that may have been affected.

## citation
If you use `cna`, please cite

[\[Reshef, Rumker\], et al., Co-varying neighborhood analysis identifies cell populations associated with phenotypes of interest from single-cell transcriptomics](https://www.nature.com/articles/s41587-021-01066-4). \[...\] contributed equally
