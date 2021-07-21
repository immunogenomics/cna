# cna
Covarying neighborhood analysis is a method for finding structure in- and conducting association analysis with multi-sample single-cell datasets. `cna` does not require a pre-specified transcriptional structure such as a clustering of the cells in the dataset. It aims instead to flexibly identify differences of all kinds between samples. `cna` is fast, does not require parameter tuning, produces measures of statistical significance for its association analyses, and allows for covariate correction.

## installation
To use `cna`, you can either install it directly from the [Python Package Index](https://pypi.org/) by running, e.g.,

`pip install cna`

or if you'd like to manipulate the source code you can clone this repository and add it to your `PYTHONPATH`.

## demo
Take a look at our [tutorial](https://nbviewer.jupyter.org/github/yakirr/cna/blob/master/demo/demo.ipynb) to see how to get started with a small synthetic data set.

## citation
If you use `cna`, please cite

[\[Reshef, Rumker\], et al., Axes of inter-sample variability among transcriptional neighborhoods reveal disease-associated cell states in single-cell data. BioRxiv, 2021](https://www.biorxiv.org/content/10.1101/2021.04.19.440534v1). \[...\] contributed equally
