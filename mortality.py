'''

Marginal impact of temperature on mortality 

Values are annual/daily expected damage resolved to GCP hierid/country level region. 

'''

import os
import click
import pprint
import logging
import datafs
import time

import itertools


from jrnr import slurm_runner

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger('uploder')
logger.setLevel('DEBUG')


__author__ = 'Justin Simcock'
__contact__ = 'jsimcock@rhg.com'
__version__ = '0.1.1'



ANNUAL_CLIMATE_FILE = (
    '/global/scratch/mdelgado/projection/gcp/climate/hierid/popwt/daily/' +
    'tas{poly}/{scenario}/{model}/{year}/1.5.nc4')

CLIMATE_COVAR = ('/global/scratch/jsimcock/data_files/covars/climate/hierid/popwt/tas_kernel_30/' +
    '{model}/{year}/0.1.1.nc4')

GDP_COVAR = ('/global/scratch/jsimcock/data_files/covars/ssp_kernel_13_gdppc/{ssp}/{econ_model}/{year}/0.1.0.nc')

GAMMAS_FILE = ('/global/scratch/jsimcock/data_files/covars/' + 
                'global_interaction_Tmean-POLY-4-AgeSpec.csvv')

WRITE_PATH = (
    '/global/scratch/jsimcock/gcp/impacts/mortality/{seed}/{scenario}/{econ_model}/{ssp}/{model}/{year}/{version}.nc')

description = '\n\n'.join(
        map(lambda s: ' '.join(s.split('\n')),
            __doc__.strip().split('\n\n')))

oneline = description.split('\n')[0]


ADDITIONAL_METADATA = dict(    
    oneline=oneline,
    description=description,
    author=__author__,
    contact=__contact__,
    version=__version__,
    repo='https://github.com/ClimateImpactLab/mortality',
    file=str(__file__),
    execute='python mortality.py run',
    project='gcp', 
    team='impacts-mortality',
    frequency='daily',
    variable='mortality-daily',
    dependencies= str([GDP_COVAR, GAMMAS_FILE,CLIMATE_COVAR, ANNUAL_CLIMATE_FILE]),
    )


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
    'NorESM1-M'
    ]))

SEED = [dict(seed=i) for i in range(1,11)]

PERIODS = [ dict(scenario='historical', year=y) for y in range(1981, 2006)] + [dict(scenario='rcp85', year=y) for y in range(2006, 2100)]

SSP = [dict(ssp='SSP' + str(i)) for i in range(1,6)]

ECONMODEL = [dict(econ_model='low'), dict(econ_model='high')]

POWER = [dict]

#we want to do a realization of all models for the periods at a given set of periods
JOB_SPEC = [PERIODS, MODELS, SSP, ECONMODEL, SEED]

@slurm_runner(filepath=__file__, job_spec=JOB_SPEC)
def mortality_annual(
                    metadata,
                    econ_model, 
                    model,
                    scenario,
                    seed, 
                    ssp, 
                    year, 
                    interactive=False):
    '''
    Calculates the IR level daily/annual effect of temperature on Mortality Rates

    Paramaters
    ----------

    climate_covar: str
        path to climate covar

    gammas_path: str
        path to csvv

    annual_climate_path: str
        path for a given year

    gdp_data: str
        path to gdp data

    year: int
        year of impacts to compute

    Returns
    -------

    Xarray Dataset 

    '''
    print(metadata)
    t1 = time.time()
    import xarray as xr
    import pandas as pd
    import numpy as np

    from impact_toolbox import (
        compute_betas,
        get_annual_climate,
        )

    metadata.update(ADDITIONAL_METADATA)
    metadata['seed'] = seed
    metadata['year'] = year
    metadata['scenario'] = scenario
    metadata['econ_model'] = econ_model
    metadata['model'] = model
    metadata['ssp'] = ssp

    if year < 2010:
        gdp_covar_path = GDP_COVAR.format(ssp=ssp, econ_model=econ_model, model=model, year=2010)

    else:
        gdp_covar_path = GDP_COVAR.format(**metadata)

    clim_covar_path = CLIMATE_COVAR.format(**metadata)

    annual_climate_paths = ANNUAL_CLIMATE_FILE.format(poly='{poly}', 
                                                    scenario=scenario, 
                                                    model=model, 
                                                    year=year)

    betas = compute_betas(clim_covar_path,gdp_covar_path, GAMMAS_FILE, ssp, econ_model, seed)

    logger.debug('reading covariate data from {}'.format(clim_covar_path))
    logger.debug('reading covariate data from {}'.format(gdp_covar_path))

    climate = get_annual_climate(annual_climate_paths, 4)

    logger.debug('reading annual weather data from {}'.format(annual_climate_paths))

    print(betas)
    print(climate)

    impact = xr.Dataset()
    
    impact['mortality_impact'] = (betas['tas']*climate['tas'] + 
                                betas['tas-poly-2']*climate['tas-poly-2'] + 
                                betas['tas-poly-3']*climate['tas-poly-3'] + 
                                betas['tas-poly-4']*climate['tas-poly-4'])

    logger.debug('Computing impact for {}'.format(year))


    impact = impact.sum(dim='time')
    impact.attrs.update({k:str(v) for k,v in metadata.items()})

    logger.debug('Computing impact for {}'.format(year))


    # write_path = WRITE_PATH(**metadata)

    # if not os.path.isdir(os.path.dirname(write_path)):
    #       os.makedirs(os.path.dirname(write_path))
        
    # impact.to_netcdf(write_path)
    t2 = time.time()
    print('Computed impact time {} for year {}'.format(t2 - t1, year))

    print(impact)

