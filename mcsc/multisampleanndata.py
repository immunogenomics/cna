import anndata as ad
import scanpy as sc
import numpy as np

class MultiSampleAnnData(ad.AnnData):
    #TODO: add __init__ that checks that samplem and d.obs.id have matching information
    #TODO create an attribute that allows access to the sample id column of data.obs

    @property
    def samplem(self):
        return self.uns['sampleXmeta']

    @samplem.setter
    def samplem(self, value):
        self.uns['sampleXmeta'] = value

    @samplem.deleter
    def samplem(self):
        del self.uns['sampleXmeta']

    def cell_to_sample(self, columns, aggregate=np.mean, sampleid='id'):
        if type(columns) == str:
            columns = [columns]
        for c in columns:
            self.samplem[c] = \
                self.obs[[sampleid, c]].groupby(by=sampleid).aggregate(aggregate)

def read(filename, **kwargs):
    return MultiSampleAnnData(sc.read(filename, **kwargs))
