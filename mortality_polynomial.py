'''

Marginal impact of temperature on mortality 

Values are annual/daily expected damage resolved to GCP hierid/country level region. 

'''

import os
import click
import pprint
import logging
import xarray as xr
import pandas as pd
import datafs


import utils
from impact_toolbox import (
        compute_betas,
        get_annual_climate,
        )

from jrnr import slurm_runner

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger('uploder')
logger.setLevel('DEBUG')


__author__ = 'Justin Simcock'
__contact__ = 'jsimcock@rhg.com'
__version__ = '0.1.0'



CLIMATE_FILE = (
    '/global/scratch/mdelgado/projection/gcp/climate/hierid/popwt/daily/' +
    'tas{poly}/{scenario}/{model}/{year}/1.5.nc4')

GDP_FILE = ('/global/scratch/jsimcock/data_files/gdppc-merged-baseline.csv')

GAMMAS_FILE = ('/global/scratch/jsimcock/data_files/covars/' + 
                'global_interaction_Tmean-POLY-4-AgeSpec.csvv')

BASELINE_CLIMATE = ('/global/scratch/mdelgado/projection/gcp/climate/hierid/popwt/daily/' +
    'tas/rcp85/{model}/2015/1.5.nc4')

WRITE_PATH = (
    '/global/scratch/jsimcock/gcp/impacts/mortality/{scenario}/{econ_model}/{ssp}/{model}/{year}/1.0.nc')

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
    repo='https://github.com/ClimateImpactLab/ceci-nest-pas-une-pipe',
    file='/mortality_polynomial.py',
    execute='python mortality_polynomial.py run',
    project='gcp', 
    team='impacts-mortality',
    frequency='daily',
    variable='mortality-daily',
    dependencies= [GDP_FILE, GAMMAS_FILE],
    pval= [0.5]
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



PERIODS = [dict(scenario='historical', year=y) for y in range(1981, 2006)] + [dict(scenario='rcp85', year=y) for y in range(2006, 2100)]

SSP = [dict(ssp='SSP' + str(i)) for i in range(1,6)]

ECONMODEL = [dict(econ_model='low'), dict(econ_model='high')]

#we want to do a realization of all models for the periods at a given set of periods
JOB_SPEC = [PERIODS, MODELS, SSP, ECONMODEL]


def mortality_annual(gammas_path, baseline_climate_path, gdp_data_path, ssp, econ_model,annual_climate_paths, write_path, year=None):
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


    betas = compute_betas(baseline_climate_path,gdp_data_path, gammas_path, ssp, econ_model)
    climate = get_annual_climate(annual_climate_paths,year, 4)

    print(betas)
    print(climate)

    impact = xr.Dataset()
    
    impact['mortality_impact'] = (betas['tas']*climate['tas'] + betas['tas-poly-2']*climate['tas-poly-2'] + 
            betas['tas-poly-3']*climate['tas-poly-3'] + betas['tas-poly-4']*climate['tas-poly-4'])

    impact = impact.sum(dim='time')

    return impact

@slurm_runner(filepath=__file__,job_spec=JOB_SPEC)
def run_job(metadata,
            model,
            year, 
            scenario,
            econ_model,
            ssp, 
            interactive=False):


    
    write_file = WRITE_PATH.format(scenario=scenario, econ_model=econ_model, ssp=ssp, model=model, year=year)
    annual_climate_paths = CLIMATE_FILE.format(poly='{poly}', scenario=scenario, model=model, year=year)
    baseline_climate_path = BASELINE_CLIMATE.format(model=model)


    if os.path.isfile(write_file):
        return


    logger.debug('calculating impact for {} {} {} {} {} '.format(scenario, model, ssp, econ_model, year))
    impact_ds = mortality_annual(GAMMAS_FILE, baseline_climate_path, GDP_FILE, ssp, econ_model, annual_climate_paths, write_file, year)

    logger.debug('udpate metadata for impact calculation {} {} {} '.format(scenario,model, year))
    impact_ds.attrs.update(ADDITIONAL_METADATA)


    # Write output
    logger.debug('attempting to write to file: {}'.format(write_file))
    if not os.path.isdir(os.path.dirname(write_file)):
        os.makedirs(os.path.dirname(write_file))
    

    if interactive:
        return impact_ds
    impact_ds.to_netcdf(write_file + '~')

    os.rename(write_file+'~', write_file)






if __name__ == '__main__':
    run_job()
    










