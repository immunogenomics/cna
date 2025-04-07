import numpy as np
import pandas as pd

def obs_to_sample(d, columns, sid_name, aggregate='mean'):
    if type(columns) == str:
        columns = [columns]
    
    samplem = pd.DataFrame(index=d.obs[sid_name].unique())
    samplem[columns] = \
        d.obs.groupby(by=sid_name)[columns].aggregate(aggregate)
    
    return samplem