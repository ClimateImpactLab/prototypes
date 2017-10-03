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
__version__ = '0.1.2'



ANNUAL_WEATHER_FILE_READ = (
    '/global/scratch/mdelgado/projection/gcp/climate/hierid/popwt/daily/' +
    '{pred}/{read_scenario}/{model}/{year}/1.5.nc4')

CLIMATE_COVAR = ('/global/scratch/jsimcock/data_files/covars/climate/hierid/popwt/tas_kernel_30/' +
    '{write_scenario}/{model}/{year}/0.1.1.nc4')

NO_ADAP_CLIM_COVAR = ('/global/scratch/jsimcock/data_files/covars/climate/hierid/popwt/tas_kernel_30/' +
    '{write_scenario}/{model}/{base_year}/0.1.1.nc4')

NO_ADAP_GDP_COVAR = ('/global/scratch/jsimcock/data_files/covars/ssp_kernel_13_gdppc/' +
    '{ssp}/{econ_model}/{base_year}/0.1.2.nc')

GDP_COVAR = ('/global/scratch/jsimcock/data_files/covars/ssp_kernel_13_gdppc/{ssp}/{econ_model}/{year}/0.1.3.nc')


GAMMAS_FILE = ('/global/scratch/jsimcock/data_files/covars/' + 
                'global_interaction_Tmean-POLY-4-AgeSpec.csvv')

T_STAR_PATH = ('/global/scratch/jsimcock/data/covars/t_star/{seed}/{scenario}/{econ_model}/{ssp}/{adaptation}.nc') 

BASE_YEAR = 2015

WRITE_PATH = ('/global/scratch/jsimcock/gcp/impacts/{variable}/{seed}/{write_scenario}/{econ_model}/{ssp}/{model}/{year}/{version}.nc')



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
    execute='python {} run'.format(str(__file__)),
    project='gcp', 
    team='impacts-mortality',
    weather_frequency='daily',
    variable='mortality-daily',
    created= str(datetime.datetime.now())
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

PERIODS = [ dict(scenario='rcp45', year=y) for y in range(1981, 2100)] + [dict(scenario='rcp85', year=y) for y in range(1981, 2100)]

SSP = [dict(ssp='SSP' + str(i)) for i in range(1,2)]

ECONMODEL = [dict(econ_model='low'), dict(econ_model='high')]


#we want to do a realization of all models for the periods at a given set of periods
JOB_SPEC = [PERIODS, MODELS, SSP, ECONMODEL]

@slurm_runner(filepath=__file__, job_spec=JOB_SPEC)
def impact_annual(
                metadata,
                econ_model, 
                model,
                scenario,
                ssp, 
                year,
                mc=False,
                interactive=False):
    '''
    Calculates the IR level daily/annual effect of temperature on Mortality Rates

    Parameters
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
    from csvv import get_gammas
    from impact import PolynomialImpact, construct_weather, construct_covars


    metadata.update(ADDITIONAL_METADATA)
    metadata['write_scenario'] = scenario
    metadata['econ_model'] = econ_model
    metadata['model'] = model
    metadata['ssp'] = ssp
    metadata['year'] = year
    metadata['base_year'] = BASE_YEAR
 
    #initialize gamma object
    gammas = get_gammas(GAMMAS_FILE)

    #Setup Run args


    #grossness
    if year <= 2005:
        metadata['read_scenario'] = 'historical'

    else:
        metadata['read_scenario'] = scenario




    #covar_setup
    #no adaptation
    no_adap_covars = construct_covars({'tas': NO_ADAP_CLIM_COVAR.format(**metadata), 'loggdppc': NO_ADAP_GDP_COVAR.format(**metadata)})
    #inconme only adaptation covars
    inc_adp_covars = construct_covars({'tas': NO_ADAP_CLIM_COVAR.format(**metadata), 'loggdppc': GDP_COVAR.format(**metadata) })
    #climate only no income adaptation
    no_inc_adp_covars = construct_covars({'tas': CLIMATE_COVAR.format(**metadata), 'loggdppc': NO_ADAP_CLIM_COVAR.format(**metadata)})
    #full_adaptation_covars
    full_adp_covars = construct_covars({'tas': CLIMATE_COVAR.format(**metadata), 'loggdppc': GDP_COVAR.format(**metadata)})
    
    #get weather for all scenarios
    weathers_paths = {k:str(ANNUAL_WEATHER_FILE_READ) for k in gammas_median.prednames.values}
    annual_weather = construct_weather(weathers_paths, metadata)

    #initialize the impact
    impact = PolynomialImpact()

    ##################
    # Compute Median #
    ##################

    #initialize median gamma values
    gammas_median = gammas.median()

    median_ds = xr.Dataset()
    
    #set metadata
    metadata['seed'] = 'median'


    #################
    # No Adaptation #
    #################
    t1 = time.time()

    betas_no_adap = (gammas_median*no_adap_covars).sum(dim='covarnames')
    t_star_no_adap = impact.get_t_star(betas_no_adap, bounds, bounds = [10,25], t_star_path =T_STAR_PATH.format(adaptation='no_adaptation', **metadata))
    no_adaptation= impact.compute(annual_weather, betas_no_adap, t_star=t_star_no_adap)
    median_ds['no_adaptation'] =  no_adaptation

    t2 = time.time()
    logger.debug('Computing median no adaptiation impact for year {}: {}'.format(year, t2-t1))

    #####################
    # Income Adaptation #
    #####################
    t1 = time.time()

    betas_income_adap = (gammas_median*inc_adp_covars).sum(dim='covarnames')
    t_star_income_adap = impact.get_t_star(betas_income_adap, bounds=[10,25], t_star_path=T_STAR_PATH.format(adaptation='income_adaptation', **metadata))
    income_adaptation = impact.compute(annual_weather, betas_income_adap, t_star=t_star_income_adap)
    median_ds['income_adaptation'] =  income_adaptation

    t2 = time.time()
    logger.debug('Computing median income adaptiation impact for year {}: {}'.format(year, t2-t1))

    ########################
    # No Income Adaptation #
    ########################
    t1 = time.time()

    betas_no_income_adap = (gammas_median*no_inc_adp_covars).sum(dim='covarnames')
    t_star_no_income_adp = impact.get_t_star(betas_no_income_adap, bounds=[10,25], t_star_path=T_STAR_PATH.format(adaptation='no_income_adaptation', **metadata))
    no_income_adaptation = impact.compute(annual_weather, gammas_median, no_inc_adp_covars,t_star_path=t_star_no_income_adp) 

    median_ds['no_income_adaptation'] =   no_income_adaptation

    t2 = time.time()
    logger.debug('Computing median no income adaptiation impact for year {}: {}'.format(year, t2-t1))


    ###################
    # Full Adaptation #
    ###################
    t1 = time.time()


    betas_full_adap = (gammas_median*full_adp_covars).sum(dim='covarnames')
    t_star_full_adp = impact.get_t_star(betas_full_adap, bounds=[10,25], t_star_path=T_STAR_PATH.format(adaptation='full_adaptation', **metadata))
    full_adaptation = impact.compute(annual_weather, betas_full_adap, t_star=t_star_full_adp)
    
    median_ds['full_adaptation'] =  full_adaptation

    t2 = time.time()
    logger.debug('Computing median full adaptiation impact for year {}: {}'.format(year, t2-t1))


    #############
    # Goodmoney #
    #############
    t1 = time.time()

    goodmoney = np.maximum(median_ds['full_adaptation'], median_ds['no_income_adaptation'])
    median_ds['goodmoney'] = goodmoney

    t2 = time.time()
    logger.debug('Computing median goodmoney adaptiation impact for year {}: {}'.format(year, t2-t1))


    median_ds.attrs.update({k: str(v) for k,v in metadata.items()})

    write_path = WRITE_PATH.format(**metadata)

    if not os.path.isdir(os.path.dirname(write_path)):
                  os.makedirs(os.path.dirname(write_path))
                
    median_ds.to_netcdf(write_path)


    ##############
    # Compute MC #
    ##############

    if mc: 
        for seed in range(13):

            t_inner_1 = time.time()
            gammas_sample = gammas.sample(seed)

            metadata['seed'] = seed

            ds_mc = xr.Dataset()

            #########################
            # compute no adaptation #
            #########################

            t_noadp1 = time.time()

            betas_no_adap = (gammas_sample*no_adap_covars).sum(dim='covarnames')
            t_star_no_adap = impact.get_t_star(betas_no_adap, bounds, bounds = [10,25], t_star_path =T_STAR_PATH.format(adaptation='no_adaptation', **metadata))
            no_adaptation= impact.compute(annual_weather, betas_no_adap, t_star=t_star_no_adap)

            ds_mc['no_adaptation']  = no_adaptation

            t_noadp2 = time.time()
            logger.debug('Computing MC no adaptiaion for {}: {}'.format(year, t_noadp2 - t_noadp1))

            ##################################
            # compute income only adaptation #
            ##################################

            t_incadp1 = time.time()

            betas_income_adap = (gammas_sample*inc_adp_covars).sum(dim='covarnames')
            t_star_income_adap = impact.get_t_star(betas_income_adap, bounds=[10,25], t_star_path=T_STAR_PATH.format(adaptation='income_adaptation', **metadata))
            income_adaptation = impact.compute(annual_weather, betas_income_adap, t_star=t_star_income_adap)

            ds_mc['income_adaptation'] = income_adaptation

            t_incadp2 = time.time()
            logger.debug('Computing income only adaptiaion for {}: {}'.format(year, t_incadp2 - t_incadp1))

            ################################
            # compute no income adaptation #
            ################################

            t_noincome1 = time.time()

            betas_no_income_adap = (gammas_sample*no_inc_adp_covars).sum(dim='covarnames')
            t_star_no_income_adap = impact.get_t_star(betas_no_income_adap, bounds=[10,25], t_star_path=T_STAR_PATH.format(adaptation='no_income_adaptation', **metadata))
            no_income_adaptation = impact.compute(annual_weather, betas_no_income_adap, t_star_no_income_adap) 

            ds_mc['no_income_adaptation'] = no_income_adaptation

            t_noincome2 = time.time()
            logger.debug('Computing no income adaptiaion for {}: {}'.format(year, t_noincome2 - t_noincome1))

            ###########################
            # compute full adaptation #
            ###########################

            t_full1 = time.time()

            betas_full_adap = (gammas_median*full_adp_covars).sum(dim='covarnames')
            t_star_full_adap = impact.get_t_star(betas_full_adap, bounds=[10,25], t_star_path=T_STAR_PATH.format(adaptation='full_adaptation', **metadata))
            full_adaptation = impact.compute(annual_weather, betas_full_adap, t_star=t_star_full_adap) 

            ds_mc['full_adaptation'] = full_adaptation

            t_full2 = time.time()
            logger.debug('Computing full adaptiaion for {}: {}'.format(year, t_full2 - t_full1))

            #######################
            # computing goodmoney #
            #######################

            t_goodmoney1 = time.time()
            goodmoney = np.maximum(ds_mc['mortality_full_adaptation'], ds_mc['no_income_adaptation'])
            ds_mc['goodmoney'] = goodmoney
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
    impact_annual()