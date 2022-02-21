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

## notices
* January 20, 2022:  It has come to our attention that a bug introduced on July 16, 2021 caused `cna` to behave incorrectly for users with `anndata` version 0.7.2 or later, possibly resulting in false positive or false negative results. This bug was fixed in `cna` version 0.1.4. We strongly recommend that any users with `anndata` version 0.7.2 or later either re-clone CNA or run `pip install --upgrade cna` and re-run all analyses that may have been affected.

## citation
If you use `cna`, please cite

[\[Reshef, Rumker\], et al., Co-varying neighborhood analysis identifies cell populations associated with phenotypes of interest from single-cell transcriptomics](https://www.nature.com/articles/s41587-021-01066-4). \[...\] contributed equally
