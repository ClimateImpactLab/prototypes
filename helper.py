import xarray as xr
import numpy as np
import pandas as pd
import itertools
import datetime


from impact_toolbox import (
        get_growth_rates, 
        get_all_gdp_annuals,
        gen_gdp_baseline,
        )

__author__ = 'Justin Simcock'
__contact__ = 'jsimcock@rhg.com'
__version__ = '0.1.0'


ntlt_path = '/global/scratch/jsimcock/social/baselines/nightlight_weight_updated.nc'

baseline_gdp_path = '/global/scratch/jsimcock/social/baselines/gdppc-merged-baseline.csv'

growth_path = '/global/scratch/jsimcock/social/baselines/gdppc-growth.csv'

write_path = '/global/scratch/jsimcock/data_files/covars/ssp_gdppc/{ssp}/{model}/{year}/{version}.nc'


ssps = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']

models = ['low', 'high']


mdata = dict(
            description='annual gdppc by hierid', 
            dependencies= str([ntlt_path, growth_path, baseline_gdp_path]),
            author=__author__, 
            contact=__contact__, 
            version=__version__, 
            source= 'https://github.com/ClimateImpactLab/mortality/blob/mortality/impact_toolbox.py',
            created = str(datetime.datetime.now())
            )
if __name__ == '__main__':

  for ssp,model in itertools.product(ssps, models):
    mdata['ssp'] = ssp
    mdata['model'] = model
    
    gen_all_gdp_annuals(ntlt_path, baseline_gdp_path, growth_path, ssp, model, version, mdata, write_path)