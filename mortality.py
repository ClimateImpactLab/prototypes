'''

Marginal impact of temperature on mortality 

Values are annual/daily expected damage resolved to GCP hierid/country level region. 

'''

import os
import logging
import time
import numpy as np
import datetime



from jrnr import slurm_runner

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger('uploader')
logger.setLevel('DEBUG')


__author__ = 'Justin Simcock'
__contact__ = 'jsimcock@rhg.com'
__version__ = '0.1.1'



ANNUAL_CLIMATE_FILE = (
    '/global/scratch/mdelgado/projection/gcp/climate/hierid/popwt/daily/' +
    'tas{poly}/{scenario}/{model}/{year}/1.5.nc4')

CLIMATE_COVAR = ('/global/scratch/jsimcock/data_files/covars/climate/hierid/popwt/tas_kernel_30/' +
    '{scenario}/{model}/{year}/0.1.1.nc4')

GDP_COVAR = ('/global/scratch/jsimcock/data_files/covars/ssp_kernel_13_gdppc/{ssp}/{econ_model}/{year}/0.1.0.nc')


GDP_2015 = ('/global/scratch/jsimcock/data_files/covars/ssp_kernel_13_gdppc/{ssp}/{econ_model}/2015/0.1.0.nc')

CLIMATE_2015 = ('/global/scratch/jsimcock/data_files/covars/climate/hierid/popwt/tas_kernel_30/' +
    'rcp85/{model}/2015/0.1.1.nc4')

GAMMAS_FILE = ('/global/scratch/jsimcock/data_files/covars/' + 
                'global_interaction_Tmean-POLY-4-AgeSpec.csvv')


baseline_impacts_path = ('/global/scratch/jsimcock/gcp/impacts/mortality/{seed}/rcp85/{econ_model}/{ssp}/{model}/baseline/{version}.nc4')

base_years = [2000, 2010]

WRITE_PATH = ('/global/scratch/jsimcock/gcp/impacts/mortality/{seed}/{scenario}/{econ_model}/{ssp}/{model}/{year}/{version}.nc4')




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
    created= str(datetime.datetime.now())
    )


MODELS = list(map(lambda x: dict(model=x), [
    'ACCESS1-0',
    # 'bcc-csm1-1',
    # 'BNU-ESM',
    # 'CanESM2',
    # 'CCSM4',
    # 'CESM1-BGC',
    # 'CNRM-CM5',
    # 'CSIRO-Mk3-6-0',
    # 'GFDL-CM3',
    # 'GFDL-ESM2G',
    # 'GFDL-ESM2M',
    # 'IPSL-CM5A-LR',
    # 'IPSL-CM5A-MR',
    # 'MIROC-ESM-CHEM',
    # 'MIROC-ESM',
    # 'MIROC5',
    # 'MPI-ESM-LR',
    # 'MPI-ESM-MR',
    # 'MRI-CGCM3',
    # 'inmcm4',
    # 'NorESM1-M'
    ]))



PERIODS = [ dict(scenario='historical', year=y) for y in range(1981, 2006)] + [dict(scenario='rcp85', year=y) for y in range(2006, 2100)]

SSP = [dict(ssp='SSP' + str(i)) for i in range(1,6)]

ECONMODEL = [dict(econ_model='low'), dict(econ_model='high')]


#we want to do a realization of all models for the periods at a given set of periods
JOB_SPEC = [PERIODS, MODELS, SSP, ECONMODEL]

@slurm_runner(filepath=__file__, job_spec=JOB_SPEC)
def mortality_annual(
                    metadata,
                    econ_model, 
                    model,
                    scenario,
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
    t_outer1 = time.time()
    import xarray as xr

    from impact_toolbox import (
        precompute_baseline, 
        prep_gammas,
        get_annual_climate,
        compute_polynomial
        )

    annual_climate_paths = ANNUAL_CLIMATE_FILE.format(poly='{poly}', 
                                                        scenario=scenario, 
                                                        model=model, 
                                                        year=year)


    gdp_covar_current_path = GDP_COVAR.format(ssp=ssp, econ_model=econ_model, year=year)
    gdp_covar_2015_path = GDP_2015.format(ssp=ssp, econ_model=econ_model)

    climate_covar_current_path = CLIMATE_COVAR.format(scenario=scenario, model=model, year=year)
    climate_covar_2015_path = CLIMATE_2015.format(model=model)


    # no_adaptation_climate = CLIMATE_2010.format(**metadata)
    # no_adaptation_gdp = GDP_2010.format(**metadata)

    with xr.open_dataset(climate_covar_2015_path) as clim_covar_2015:
        clim_covar_2015.load()


    with xr.open_dataset(gdp_covar_2015_path) as gdp_covar_2015:
        gdp_covar_2015.load()



    if year < 2015:
        gdp_covar = gdp_covar_2015
        clim_covar = clim_covar_2015

    else: 
        with xr.open_dataset(gdp_covar_current_path,autoclose=True) as gdp_covar:
            gdp_covar.load()
        logger.debug('reading covariate data from {}'.format(gdp_covar_current_path))

        with xr.open_dataset(climate_covar_current_path, autoclose=True) as clim_covar:
            clim_covar.load()
        logger.debug('reading covariate data from {}'.format(climate_covar_current_path))



    t1 = time.time()
    climate = get_annual_climate(annual_climate_paths, 4)
    t2 = time.time()
    logger.debug('reading annual weather data from {}: {}'.format(annual_climate_paths, t2-t1))


    for seed in range(13):

        t_inner_1 = time.time()
        gammas = prep_gammas(GAMMAS_FILE, seed)

        metadata.update(ADDITIONAL_METADATA)
        metadata['seed'] = seed
        metadata['year'] = year
        metadata['scenario'] = scenario
        metadata['econ_model'] = econ_model
        metadata['model'] = model
        metadata['ssp'] = ssp
        
        impact = xr.Dataset()

        ####################
        # compute_baseline #
        ####################

        base_path = baseline_impacts_path.format(**metadata)

        t_base1 = time.time()
        baseline = precompute_baseline(ANNUAL_CLIMATE_FILE, gdp_covar_2015_path, climate_covar_2015_path, gammas, metadata, 2000, 2010, poly=4, write_path=base_path)
        t_base2 = time.time()
        logger.debug('Computing baseline for {} {} {} {}: {}'.format(scenario, econ_model, model, ssp, t_base2 - t_base1))

        #ßflat_curve_adaptation = find_poly_mins

        #########################
        # compute no adaptation #
        #########################

        t_noadp1 = time.time()

        no_adaptation  = compute_polynomial(climate, clim_covar_2015, gdp_covar_2015, gammas)

        impact['no_adaptation'] = (no_adaptation.sum(dim='time') - baseline['baseline'])*1e5

        t_noadp2 = time.time()
        logger.debug('Computing no adaptiaion for {}: {}'.format(year, t_noadp2 - t_noadp1))

        #############################
        # compute income adaptation #
        #############################

        t_incadp1 = time.time()

        income_adaptation = compute_polynomial(climate, clim_covar_2015, gdp_covar, gammas)

        impact['income_adaptation'] = (income_adaptation.sum(dim='time') - baseline['baseline'])*1e5

        t_incadp2 = time.time()
        logger.debug('Computing income only adaptiaion for {}: {}'.format(year, t_incadp2 - t_incadp1))

        ###########################
        # compute full adaptation #
        ###########################

        t_full1 = time.time()

        mortality_full_adaptation = compute_polynomial(climate, clim_covar, gdp_covar, gammas)

        impact['mortality_full_adaptation'] = (mortality_full_adaptation.sum(dim='time') - baseline['baseline'])/1e5

        t_full2 = time.time()
        logger.debug('Computing full adaptiaion for {}: {}'.format(year, t_full2 - t_full1))

        ################################
        # compute no income adaptation #
        ################################

        t_noincome1 = time.time()

        no_income_adaptation = compute_polynomial(climate, clim_covar, gdp_covar_2015, gammas)

        impact['no_income_adaptation'] = (no_income_adaptation.sum(dim='time') - baseline['baseline'])/1e5

        t_noincome2 = time.time()
        logger.debug('Computing no income adaptiaion for {}: {}'.format(year, t_noincome2 - t_noincome1))

        #######################
        # computing goodmoney #
        #######################

        t_goodmoney1 = time.time()
        impact['goodmoney'] = np.maximum(impact['mortality_full_adaptation'], impact['no_income_adaptation'])
        t_goodmoney2 = time.time()
        logger.debug('Computing goodmoney for {}: {}'.format(year, t_goodmoney2 - t_goodmoney1))

        logger.debug('Computing impact for {}'.format(year))
   

        impact.attrs.update({k:str(v) for k,v in metadata.items()})

        write_path = WRITE_PATH.format(**metadata)

        if not os.path.isdir(os.path.dirname(write_path)):
              os.makedirs(os.path.dirname(write_path))
            
        impact.to_netcdf(write_path)

        t_inner_2 = time.time()
        logger.debug('Inner Loop time for {}: {}'.format(year, t_inner_2 - t_inner_1))

    t_outer2 = time.time()
    logger.debug('Computed impact for year {}: {}'.format(year, t_outer2 - t_outer1))


if __name__ == '__main__':
    mortality_annual()