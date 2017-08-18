import xarray as xr
import numpy as np
import pandas as pd
import itertools
import datetime


from impact_toolbox import (
        get_growth_rates, 
        gen_all_gdp_annuals,
        gen_gdp_baseline,
        gen_kernel_covars
        )

__author__ = 'Justin Simcock'
__contact__ = 'jsimcock@rhg.com'
__version__ = '0.1.0'


# ntlt_path = '/global/scratch/jsimcock/social/baselines/nightlight_weight_updated.nc'

# baseline_gdp_path = '/global/scratch/jsimcock/social/baselines/gdppc-merged-baseline.csv'

# growth_path = '/global/scratch/jsimcock/social/baselines/gdppc-growth.csv'

write_path_brc = ('/global/scratch/jsimcock/data_files/covars/ssp_kernel_' + kernel +
                '_gdppc/{ssp}/{model}/{year}/{version}.nc')

covar_path_brc = '/global/scratch/jsimcock/data_files/covars/ssp_gdppc/{ssp}/{model}/{year}/{version}.nc'

# write_path = '/Users/justinsimcock/data/gdps/ssp_kernel_gddpc/{ssp}/{model}/{year}/{version}.nc'

# covar_path = '/Users/justinsimcock/data/gdps/ssp_gddpc_{ssp}_{model}_{year}_1.0.nc'

ssps = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']

models = ['low', 'high']

kernel = 13


mdata = dict(
            description='annual kernelized gdppc by hierid', 
            dependencies= '',
            author=__author__, 
            contact=__contact__, 
            version=__version__, 
            source= 'https://github.com/ClimateImpactLab/mortality/blob/mortality/impact_toolbox.py',
            created = str(datetime.datetime.now()),
            )
if __name__ == '__main__':

  for ssp,model in itertools.product(ssps, models):
    mdata['ssp'] = ssp
    mdata['model'] = model

    for y in range(2010, 2100):
      window = range(y-(kernel-1), y+1)
      #print(window)
      paths = [covar_path_brc.format(ssp=ssp, model=model,year=yr) for yr in window]
      #print(paths)
      mdata['dependencies'] = str(paths)
      mdata['year'] = y 

      gen_kernel_covars(paths, kernel = kernel, metadata=mdata, write_path=write_path_brc)