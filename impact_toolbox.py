'''
This file is an interface for a set of functions needed in impact forecasting

------------------
My notes and questions 

1. Mortality: Steps below do not represent order
    A. Get covariate values for GDP and Temp
            1. GDP values are some rolling mean from a base line
            2. Temp values are some rolling mean from a base line
            3. In both cases, need baseline year, method of computing mean, and mean period


    B. Get Gamma values
            1. Read in csvv with gamma
            2. Take draw from multivariate distribution to determnine gamma values at given p-value

    C. Define mathematical relationship between covariates and gammas
            1. This is us defnining the dot product or MLE spec
            2. Current implementation uses np.polyval


    D. Determine specific temperature where the minimum value is realized
            1. Evaluate the function, get the minimum   
            2. Get the minimum with np.argmin
            3. Get the min  with impact-common/impactcommon/math/minpoly.findpolymin
            4. I don't understand what he is trying to accomplish with this?

    E. Define the function under the goodmoney assumption
            1. Calculate the response under full adapation
            2. Calculate the response under no income adaption
            3. Take the best response of these two

    F. Evaluate the function under the different 'farmer' scenarios
            1. Income and weather updates
            2. Income but no weather updates
            3. No updates

    H. Something to generate the marginal income effect (climtas_effect). What is this?

    I. Some additional transformation on the curve to constrain the results
            1. Something that compares the regional baseline mins to some 
            other value (specified by climtas effect curve? )

    J. Do something that compares the results from income/weather updates 
    (farm_curvegen) and the climtas_effect (whatever that is)?
'''
import xarray as xr
import numpy as np
import pandas as pd
#import metacsv
from six import string_types
import itertools
import toolz
import time
import datafs
import csv
import os




def compute_climate_covariates(path,base_year=None, rolling_window=None):
    ''' 
    Method to calculate climate covariate

    Paramaters
    ----------
    path: str
        glob of paths to climate data

    baseline_year: int
        year from which to generate baseline value

    rolling_window: int
        num years to calculate the rolling covariate average


    .. note: Rolling window only used for full adaptation, otherwise None

    Returns
    -------
    Xarray Dataset
        daily/annual climate values 
    '''


    #compute baseline
    #load datasets up through baseline year
    #compute an average value and set baseline for climate

    #compute annuals
    #for each additional year year that you read in reset baseline value and 
    #update the next years climate with that value
    #output will be a impact_region by time dimension dataset where the value in 
    #for a given time period is the rolling mean temp. 
    #is there a way to do this without looping?

    #baseyear_path = path.format(year=base_year)



    t1 = time.time()
    ds = xr.open_dataset(path)
    #print(ds)
    ds = ds['tas'] .groupby('hierid').mean()
    t2 = time.time()
    print('Finishing compute_climate_covariates: {}'.format(t2-t1))
    return ds




def compute_gdp_covariates(path, ssp, econ_model,base_year=2010):
    '''
    Method to calculate climate covariate

    Paramaters
    ----------
    path: str
        glob of paths to GDP data

    baseline_year: int
        year from which to generate baseline value

    rolling_window: int
        num years to calculate the rolling covariate average

    year: int
        baseline year 


    Returns
    -------
    Xarray Dataset
        annual GDP values 

    '''

    #compute baseline
    t1 = time.time()
    df = pd.read_csv(path, skiprows=10)
    df = df.loc[(df['model']==econ_model) & (df['scenario'] == ssp) & (df['year'] == base_year)]
    df['value'] = np.log(df['value'])
    df = df[['hierid', 'value']]
    t2 = time.time()
    print('Completing compute_gdp_covariates: {}'.format(t2 - t1))

    return df


def gen_gdp_covariates_file(inpath, rolling_window):
    # new_df = pd.DataFrame()
    # df = pd.read_csv(inpath, skiprows=10)
    # df = df.loc[(df['model']=='low') & (df['scenario'] == 'SSP1')]
    # for year in range(2010, 2100):
    #     #get a baseline
    #     # if (year >= 2010) & (year < 2015):
    #     #     #print(year)
    #     #     df = df.loc[(df['year'] >= 2010) & (df['year'] <= 2015)]
    #     #     value = df.groupby('hierid').mean()['value']
    #     #     value= pd.DataFrame(value).reset_index()
    #     #     df1 = pd.DataFrame(value, columns=['hierid', 'value'])
    #     #     df1['year'] = year
    #     #     new_df = new_df.append(df1)
    #     #     #print(new_df.head())

    #     # # # for years in rolling window
    #     # # #For years after 2014 
    #     # if ((year - 2015) >= 0) & ((year - 2015) <= rolling_window):
    #     #     #print(year)
    #     #     df = df.loc[(df['year'] >= 2015) & (df['year'] <= year)]
    #     #     value = df.groupby('hierid').mean()['value']
    #     #     value= pd.DataFrame(value).reset_index()
    #     #     df1 = pd.DataFrame(value, columns=['hierid', 'value'])
    #     #     df1['year'] = year
    #     #     new_df = new_df.append(df1)
    #     #     #print(len(new_df))
    #     #     #print(new_df.head())

    #     if year > 2030:
    #         gap = year-rolling_window
    #         print(year, gap)

    #         df = df.loc[(df['year'] >= gap) & (df['year'] <= year)]
    #         print(df['year'].min(), df['year'].max())
    #         value = df.groupby('hierid').mean()['value']
    #         value= pd.DataFrame(value).reset_index()
    #         df1 = pd.DataFrame(value, columns=['hierid', 'value'])
    #         df1['year'] = year
    #         new_df = new_df.append(df1)
    #             #print(new_df.head())
    #         print(len(new_df))
    pass


def gen_gdp_baseline(nightlights, gdp_baseline_file, ssp, econ_model, base_year=2010):
    '''
    Function to generate the baseline gdp values

    1. load nightlights file (228)
    2. Load baseline gdppc file (230)
    3. Select an ssp, model and base_year (233)
    4. merge the two dataframes on hierid (236)
    5. create a list of gdppc_ratio weight min values based on iso grouping (241)
    6. create a mask and replace 0.0 gdppc_ratio values with min for that iso (244)
    7. replace 0.0 values in merged dataframe with masked values (247)
    8. multiply ratios by gdppc_values to get baselines by hierid (252)

    Parameters:
    nightlights: str
        path to nightlights weight dataset

    gdp_baseline_file: str
        path to baseline gdp dataset

    ssp: str
        SSP values 1 through 5

    econ_model: str
        `high` or `low`
    
    base_year: int


    '''

    #load nightlights
    nlts = pd.read_csv(nightlights)
    #load baseline gdp file
    base = pd.read_csv(gdp_baseline_file, skiprows=10)

    #load baseline year, ssp, and model
    base= base.loc[(base['model']==econ_model) & (base['scenario'] == ssp) & (base['year'] == base_year)]

    #merge the two files to map hierids to gdp_ratios and gdp vals
    new = pd.merge(base, nlts, how='inner', on='hierid')
    
    #find the min ratio for a given iso that is greater than 0.0. 
    #We are setting all 0.0 vals to this val
    min_ratios = new[new['gdppc_ratio'] > 0.0].groupby('iso')['gdppc_ratio'].min()

    #create a df mask df that sets 
    new_zeros = new[new['gdppc_ratio'] ==0.0].apply(lambda x: x.replace(x['gdppc_ratio'], min_ratios[x['iso']]), 1)

    # #if there are 0.0 vals for a hierid, set the gdppc_ratio to the min val for that iso
    new.loc[new['gdppc_ratio'] == 0.0, 'gdppc_ratio'] = new_zeros['gdppc_ratio']

    test = new.apply(lambda x: x['value']*x['gdppc_ratio'], 1)

    annual_baseline = xr.Dataset({'gdppc': (['hierid'], 
                                    test.values)},
                                    coords={'hierid': new['hierid'], 'iso': new['iso']}
                                  )
    annual_baseline.attrs.update(dict(year=base_year, ssp=ssp, econ_model=econ_model))
    

    path ='/Users/justinsimcock/data/socio_covariates/gdppc/{}_{}_{}.nc'.format(econ_model, ssp,base_year)
    
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
        annual_baseline.to_netcdf(path)
    
    return annual_baseline


def gen_gdp_annual(gdp_growth_path, ssp, econ_model, year=None):
    '''
    1. Load in merged_df
    2. load in growth_df
    3. If year is % 5 multiply val in year  times growth rate and reset growth rate 
    4. else compute value of current year based on value from last year and last years growth rate

    '''
    gdf = pd.read_csv(gdp_growth_path, skiprows=9)
    gdf = gdf.loc[(gdf['model'] == econ_model) & (gdf['scenario'] == ssp) & (gdf['year'] == year)]    
    gdf_series = pd.Series(data=gdf.growth.values, index=gdf.iso)
    return gdf_series


    # baseline['estimated_gdp'] = baseline.apply(lambda x: x['baseline']*annual_growth[annual_growth['iso'
    # ...: ] ==x['iso']]['growth'])
      

def annual_gdp_to_netcdf(baseline_df, ):



  annual_ds = xr.Dataset
  mdata = zip(baseline.columns[:3], (baseline.year[0], baseline.model[0], baseline.scenario[0]))




def replace_zeros(row):
    '''
    replaces all 0.0 values in a pd Dataframe with the lowest non zero-value 
    for the ISO
    '''
    min_ratios = get_mins()

    if row['gdppc_ratio'] == 0.0:
        row['gdppc_ratio'] = min_ratios[row['iso']]



    


def read_csvv(path):
    '''
    Returns the gammas and covariance matrix 

    Returns
    -------
    dict 
    '''

    t1 = time.time()
    data = {}

    #constant, climtas, gdp


    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'gamma':
                data['gamma'] = reader.next()
            if row[0] == 'gammavcv':
                data['gammavcv'] = reader.next()
            if row[0] == 'residvcv':
                data['residvcv'] = reader.next()

    t2 = time.time()
    print('Completing read_csvv:{}'.format(t2 - t1))
    return data['gamma']

def prep_gammas(path):
    '''
    Randomly draws gammas from a multivariate distribution

    Parameters
    ----------
    path: str
        path to file 

    seed: int
        seed for random draw

    Returns
    -------
    dict
    '''
    t1 = time.time()
    indices = {'age_cohorts': pd.Index(['infant', 'mid', 'advanced'], name='age')}

    data = [float(num) for num in read_csvv(path)]
    gammas = xr.Dataset()


    for pwr in range(1,5):
            gammas['beta0_pow{}'.format(pwr)] = xr.DataArray(
                data[pwr-1::12], dims=('age',), coords={'age':indices['age_cohorts']})
            gammas['gdp_pow{}'.format(pwr)] = xr.DataArray(
                data[pwr::12], dims=('age',), coords={'age': indices['age_cohorts']})
            gammas['tavg_pow{}'.format(pwr)] = xr.DataArray(
                data[pwr+1::12], dims=('age',), coords={'age': indices['age_cohorts']})

    t2 = time.time()
    print('completing prep_gammas: {}'.format(t2- t1))
    return gammas
    

def prep_covars(gdp_path, clim_path, ssp, econ_model, base_year=None):

    t1 = time.time()
    covars = xr.Dataset()

    gdp = compute_gdp_covariates(gdp_path, ssp, econ_model, base_year=2010)
    tas_avg = compute_climate_covariates(clim_path)
    covars['gdp'] = xr.DataArray(gdp['value'], dims=('hierid'), coords={'hierid':gdp['hierid']})
    covars['tavg'] = tas_avg
    t2 = time.time()
    print('completing prep_covars:{}'.format(t2- t1))
    return covars



def compute_betas(clim_path, gdp_path, gammas_path, ssp, econ_model):
    '''
    Computes the matrices beta*gamma x IR for each covariates 

    1. Calls method to get gammas at given p-value
    2. Calls method toompute gdp covariate
    3. Calls method to compute tas covariate
    4. Computes outer product of 

    Parameters
    ----------


    Returns
    -------

    3 arrays representing 


    '''

    t1 = time.time()
    covars = prep_covars(gdp_path, clim_path, ssp, econ_model)
    gammas = prep_gammas(gammas_path)



    betas = xr.Dataset()

    betas['tas'] = (gammas['beta0_pow1'] + gammas['gdp_pow1'] * covars['gdp'] + gammas['tavg_pow1']*covars['tavg'])
    print(betas['tas'])
    betas['tas-poly-2'] = (gammas['beta0_pow2'] + gammas['gdp_pow2'] * covars['gdp'] + gammas['tavg_pow2']*covars['tavg'])
    print(betas['tas-poly-2'])
    betas['tas-poly-3'] = (gammas['beta0_pow3'] + gammas['gdp_pow3'] * covars['gdp'] + gammas['tavg_pow3']*covars['tavg'])
    print(betas['tas-poly-3'])
    betas['tas-poly-4'] = (gammas['beta0_pow4'] + gammas['gdp_pow4'] * covars['gdp'] + gammas['tavg_pow4']*covars['tavg'])
    print(betas['tas-poly-4'])



    t2 = time.time()
    print('Completing compute_betas: {}'.format(t2 - t1))
    return betas

def get_annual_climate(model_paths, year, polymomial):
    '''
    models_paths: list
        list of strings to temperature variables for that year

    year: str 

    '''
    t1 =time.time()
    print(model_paths.format(poly='',year=year))

    dataset = xr.Dataset()

    with xr.open_dataset(model_paths.format(poly='',year=year)) as ds:
        ds.load()

    varname = ds.variable
    dataset[varname] = ds[varname]
    

    for poly in range(2, polymomial+1):
        fp = model_paths.format(poly='-poly-{}'.format(poly),year=year)

        with xr.open_dataset(fp) as ds:
            ds.load()

        varname = ds.variable
        print(varname)
        print(ds)
        dataset[varname] = ds[varname]
    t2 = time.time()
    print('get_climate_paths: {}'.format(t2 -t1))
    return dataset

# def mortality_annual(gammas_path, baseline_climate_path, gdp_data_path, annual_climate_paths, write_path, year=None):
#     '''
#     Calculates the IR level daily/annual effect of temperature on Mortality Rates

#     Paramaters
#     ----------
#     gammas_path: str
#         path to csvv

#     climate_data: str
#         path to baseline year climate dataset

#     gdp_data: str
#         path to gdp_dataset

#     annual_climate_paths: list
#         list of paths for climate data sets

#     mortality_flags: dit
#         set of methods optionally applied to adjust final impact 


#     Returns
#     -------

#     Xarray Dataset 


#     '''
#     metadata = dict(
#             dependencies = [gammas_path, baseline_climate_path, gdp_data_path],
#             description ='median mortality run on ACCESS1-0 model',
#             year=year
#             )

#     betas = compute_betas(baseline_climate_path,gdp_data_path, gammas_path)
#     climate = get_annual_climate(annual_climate_paths,year, 4)
#     write_path = write_path.format(year=year)

#     impact = xr.Dataset()
    
#     impact['mortality_impact'] = (betas['tas']*climate['tas'] + betas['tas-poly-2']*climate['tas-poly-2'] + 
#             betas['tas-poly-3']*climate['tas-poly-3'] + betas['tas-poly-3']*climate['tas-poly-3'])

#     impact.attrs.update(metadata)
#     if not os.path.isdir(os.path.dirname(write_path)):
#         os.makedirs(os.path.dirname(write_path))

#     impact.to_netcdf(write_path)
    
#     print('Writing {}'.format(year))




def pval_thing():
    '''
    Generate a list of pvals
    '''


def goodmoney(ds):
    '''
    Some method to transform data according to goodmoney
    '''


def combine(ds):
    '''
    If we are doing age cohorts, sums the damages for each IR across age cohorts
    '''
    pass


def costs(ds, *args):
    '''
    Some methods to account for costs
    '''
    pass
