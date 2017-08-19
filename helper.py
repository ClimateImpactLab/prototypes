
import os
import logging
import datetime
import itertools
import numpy as np
import pandas as pd
import xarray as xr

from jrnr import slurm_runner


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger('uploader')
logger.setLevel('DEBUG')


__author__ = 'Justin Simcock'
__contact__ = 'jsimcock@rhg.com'
__version__ = '0.1.0'

write_path_brc = ('/global/scratch/jsimcock/data_files/covars/climate/hierid/popwt/{variable}_kernel_{kernel}'
                + '/{rcp}/{model}/{year}/{version}.nc')

covar_path_brc = ('/global/scratch/mdelgado/projection/gcp/climate/hierid/popwt/daily/{variable}/{rcp}/{model}/{year}/1.5.nc')



ADDITIONAL_METADATA = dict(
            description='annual kernelized gdppc by hierid', 
            dependencies= '',
            author=__author__, 
            contact=__contact__, 
            version=__version__, 
            source= str(__file__),
            project='gcp',
            created = str(datetime.datetime.now())
            )


KER = [dict(kernel='30')]

VAR = [dict(variable='tas')]

MODELS = list(map(lambda x: dict(model=x), [
    'ACCESS1-0',
    'bcc-csm1-1',
    'BNU-ESM',
    'CanESM2',
    'CCSM4',
    'CESM1-BGC',
    'CNRM-CM5',
    'CSIRO-Mk3-6-0',
    'GFDL-CM3',
    'GFDL-ESM2G',
    'GFDL-ESM2M',
    'IPSL-CM5A-LR',
    'IPSL-CM5A-MR',
    'MIROC-ESM-CHEM',
    'MIROC-ESM',
    'MIROC5',
    'MPI-ESM-LR',
    'MPI-ESM-MR',
    'MRI-CGCM3',
    'inmcm4',
    'NorESM1-M']))

JOB_SPEC = [KER, VAR,  MODELS]


@slurm_runner(filepath=__file__, job_spec=JOB_SPEC, onfinish=onfinish)
def gen_covars(
          mdata,
          model, 
          variable, 
          kernel
            )

  from impact_toolbox import (gen_kernel_covars)


    for y in range(1981, 2100):
      window = range(y-(kernel-1), y+1)
      #print(window)
      paths = []
      for yr in window:
        rcp == 'historical'
        if yr > 2005:
          rcp == 'rcp85'
        paths.append(covar_path_brc.format(rcp=rcp, model=model,variable=variable, year=yr) for yr in window)
      #print(paths)
      mdata.update(ADDITIONAL_METADATA)
      mdata['dependencies'] = str(paths)
      mdata['year'] = y

      write_path = write_path_brc.format(**mdata)

      gen_kernel_covars(paths, climate=True, metadata=mdata, write_path=write_path)