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

#import metacsv
import os
import re
import csv
import glob
import time
import toolz
import datafs
import datetime
import itertools
import xarray as xr
import numpy as np
import pandas as pd
from six import string_types
from scipy.stats import multivariate_normal as mn



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


def gen_gdp_baseline(nightlights_path, gdp_baseline_file, ssp, model, base_year=2010, metadata=None, write_path=None):
    '''
    Function to generate the baseline gdp values. Generates baseline by ssp and model for baseline year 2010. 
    If write_path is given will write to disk. 

    1. load nightlights netcdf file 
    2. Load baseline gdppc csv file as xr Dataset
    3. Construct data structure for the baseline 
    4. Update Metadata
    5. Do Math
    6. Select model, ssp and base_year
    6. Update NaN's with global mean from dataset
    7. Write to disk

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
      baseline year to start gdp calculations

    write_path: str
      path to write outputs

    metadata: dict
      details for file generation process
    '''

    #load nightlights
    ntlt = xr.open_dataset(nightlights_path)

    #load baseline gdp file
    base = xr.Dataset.from_dataframe(pd.read_csv(gdp_baseline_file, skiprows=10, index_col=range(4))).sel(year=base_year).drop('year')

    #create the data structure for the product of ntlt and base
    product = xr.Dataset(coords={'hierid': ntlt.hierid, 'model': base.model, 'scenario': base.scenario, 'iso':ntlt.iso})

    #do math: generates baseline numbers for all scenarios and models
    product['gdppc'] = base['value'] * ntlt['gdppc_ratio']


    #slice to select a specific model and scenario
    product = product.sel(scenario=ssp, model=model)

    #fillna's for this model
    product['gdppc'] = product.gdppc.fillna(product.gdppc.mean())

    #update metadata
    if metadata:
      product.attrs.update(metadata)

    #write to disk
    if write_path:
      if not os.path.isdir(os.path.dirname(write_path)):
          os.makedirs(os.path.dirname(write_path))

      product.to_netcdf(write_path)
    
    return product

def gen_nightlights_netcdf(nightlights_path, metadata, write_path):
    '''
    Helper function to convert nightlight csv file to netcdf. 
    Does some data cleaning and filling in values where necessary

    Parameters
    ----------
    nightlights_path: str
        path to nightlights csv

    metadata: dict

    write_path: str
        file to save to 

    1. Find iso-level min 
    2. Assign all 0.0 values to that iso's min value
    3. Fillna's with 1.0
    4. write metadata
    4. convert to netcdf
    '''

    ntlt =  pd.read_csv(nightlights_path, index_col=0)

    min_ratios = ntlt[ntlt['gdppc_ratio'] > 0.0].groupby('iso')['gdppc_ratio'].min()

    fill_zeros = ntlt[ntlt['gdppc_ratio'] == 0.0].apply(lambda x: x.replace(x['gdppc_ratio'], min_ratios[x['iso']]), 1)

    ntlt.loc[ntlt['gdppc_ratio'] == 0.0, 'gdppc_ratio'] = fill_zeros['gdppc_ratio']

    ntlt = ntlt.fillna(1.0)
    ntlt = xr.Dataset.from_dataframe(ntlt).set_coords('iso')

    if not os.path.isdir(os.path.dirname(write_path)):
        os.makedirs(os.path.dirname(write_path))


    ntlt.attrs.update(metadata)
    ntlt.to_netcdf(write_path)

    return ntlt

def compute_annual(previous_year_ds, growth_ds, write_path=None, metadata=None):
    '''
    Computes annual gdp based on last years gdp and the growth rate
    Simple multiplication

    Parameters
    ----------
    previous_year_ds: xarray Dataset
        last years gdp growth for each hierid

    growth_ds: Xarray Dataset
        dataset of ISO by growth_rate

    write_path: str
        file path to write to

    metadata: dict
        file specific metadata

    Examples
    --------

    In [1]: annual = compute_annual(annual_2013, growth, write_path='annual_2014_gdp.nc', metadata=metadata)
    Out[1]:
    <xarray.Dataset>
    Dimensions:   (hierid: 24378, iso: 185)
    Coordinates:
        year      int64 2014
        model     |S3 'low'
        scenario  |S4 'SSP1'
        * iso       (iso) object 'ABW' 'AFG' 'AGO' 'ALB' 'ARE' 'ARG' 'ARM' 'AUS' ...
        * hierid    (hierid) object 'ABW' 'AFG.11.R888b226f710b3709' 'AFG.15.167' ...
    Data variables:
        gdppc     (iso, hierid) float64 1.745e+04 1.745e+04 1.745e+04 1.745e+04 ...
    '''

    annual = xr.Dataset()
    annual['gdppc'] = growth_ds['growth']*previous_year_ds['gdppc']

    if metadata:
        annual.attrs.update(metadata)

    if write_path:
      if not os.path.isdir(os.path.dirname(write_path)):
          os.makedirs(os.path.dirname(write_path))

      annual.to_netcdf(write_path)


    return annual

def get_growth_rates(gdp_growth_path):
    '''

    Parameters
    ----------
    gdp_growth_path:str

    ssp: str
    one of the following: 'SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5',

    econ_model: str
    one of the following: 'high', 'low'

    year: int
    year to get gdp for growth rate

    Returns
    -------
    Xarray Dataset of iso x gdppc for a given year

    '''

    growth_df = pd.read_csv(gdp_growth_path, skiprows=10).drop_duplicates()
    growth = xr.Dataset.from_dataframe(growth_df.set_index(list(growth_df.columns[:4])))

    #for locations that do not have a growth rate, supply the global mean value
    
    return growth  


def read_csvv(path):
    '''
    Returns the gammas and covariance matrix 
    
    Parameters
    ----------
    path: str
        path to csvv file

    Returns
    -------
    dict with keys of gamma, gammavcv, and residvcv
    '''

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



    return data

def prep_gammas(path, power=4, seed=None):
    '''
    Randomly draws gammas from a multivariate distribution

    Parameters
    ----------
    path: str
        path to file 

    power: int
      for polynomial specifications

    seed: int
        seed for random draw

    Returns
    -------
    Xarray Dataset
    '''

    ##########################
    # Read in data from csvv #
    ##########################
    covar_data = read_csvv(path)

    ##########################################
    # convert gamma and covariance to floats #
    ##########################################
    _gamma = [float(num) for num in covar_data['gamma']]
    _covar = [float(num) for num in covar_data['gammavcv']]

    if seed: 
      np.random.seed(seed)

    data = mn.rvs(_gamma, _covar)


    gammas = xr.Dataset()
    indices = {'age_cohorts': pd.Index(['age0-4', 'age5-64', 'age65+'], name='age')}


    for pwr in range(1,power+1):
            gammas['beta0_pow{}'.format(pwr)] = xr.DataArray(
                data[pwr-1::12], dims=('age',), coords={'age':indices['age_cohorts']})
            gammas['gdp_pow{}'.format(pwr)] = xr.DataArray(
                data[pwr::12], dims=('age',), coords={'age': indices['age_cohorts']})
            gammas['tavg_pow{}'.format(pwr)] = xr.DataArray(
                data[pwr+1::12], dims=('age',), coords={'age': indices['age_cohorts']})

    return gammas
    


##################
#Will deprecate or refactor
def prep_covars(gdp_path, clim_path, ssp, econ_model, base_year=None):
    '''
    Loads and aligns covar data sets

    Parameters
    ----------
    gdp_path: str
        path to a given year of gdp covariate data

    clim_path: str
        path to a given year of climate covariate data

    ssp: str
        one of the following: 'SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5',

    econ_model: str
        one of the following: 'high', 'low'

    base_year: int
        base year to use as baseline for calcualtions

    Returns
    -------
    Xarray dataset with gdp and tavg covariate variables
    '''

    t1 = time.time()
    covars = xr.Dataset()

    gdp = compute_gdp_covariates(gdp_path, ssp, econ_model, base_year=2010)
    tas_avg = compute_climate_covariates(clim_path)
    covars['gdp'] = xr.DataArray(gdp['value'], dims=('hierid'), coords={'hierid':gdp['hierid']})
    covars['tavg'] = tas_avg
    t2 = time.time()
    print('completing prep_covars:{}'.format(t2- t1))
    return covars

####################

def compute_betas(clim_path, gdp_path, gammas_path, seed):
    '''
    Computes the matrices beta*gamma x IR for each covariates 

    1. Calls method to get gammas at given p-value
    2. Calls method toompute gdp covariate
    3. Calls method to compute tas covariate
    4. Computes outer product of 

    Parameters
    ----------
    clim_path: str
        path to climate covariate

    gdp_path: str
        path to gdp covariate

    ssp: str
        one of the following: 'SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5',

    econ_model: str
        one of the following: 'high', 'low'
 
    Returns
    -------
    Xarray Dataset with values for each of the Betas

    '''

    t1 = time.time()
    gdp_covar = xr.open_dataset(gdp_path)
    clim_covar = xr.open_dataset(clim_path)

    gammas = prep_gammas(gammas_path, seed)



    betas = xr.Dataset()

    betas['tas'] = (gammas['beta0_pow1'] + gammas['gdp_pow1'] * gdp_covar['gdppc'] + gammas['tavg_pow1']*clim_covar['tas'])
    print(betas['tas'])
    betas['tas-poly-2'] = (gammas['beta0_pow2'] + gammas['gdp_pow2'] * gdp_covar['gdppc'] + gammas['tavg_pow2']*clim_covars['tas'])
    print(betas['tas-poly-2'])
    betas['tas-poly-3'] = (gammas['beta0_pow3'] + gammas['gdp_pow3'] * gdp_covar['gdppc'] + gammas['tavg_pow3']*clim_covars['tas'])
    print(betas['tas-poly-3'])
    betas['tas-poly-4'] = (gammas['beta0_pow4'] + gammas['gdp_pow4'] * gdp_covar['gdppc'] + gammas['tavg_pow4']*clim_covars['tas'])
    print(betas['tas-poly-4'])



    t2 = time.time()
    print('Completing compute_betas: {}'.format(t2 - t1))
    return betas

def get_annual_climate(model_paths, polymomial=4):
    '''
    Parameters
    ----------
    models_paths: list
        list of strings to temperature variables for that year

    year: str 
        specific year to calculate climate for

    polynomial: int
        order of polynomial to read
        used for path handling


    Returns
    -------
    Xarray Dataset with each climate polynomial as a variable 


    '''

    dataset = xr.Dataset()

    with xr.open_dataset(model_paths.format(poly='')) as ds:
        ds.load()

    varname = ds.variable
    dataset[varname] = ds[varname]
    

    for poly in range(2, polymomial+1):
        fp = model_paths.format(poly='-poly-{}'.format(poly))

        with xr.open_dataset(fp) as ds:
            ds.load()

        varname = ds.variable
        print(varname)
        print(ds)
        dataset[varname] = ds[varname]
    t2 = time.time()
    print('get_climate_paths: {}'.format(t2 -t1))
    return dataset


def gen_all_gdp_annuals(nightlights_path, baseline_gdp_path, growth_path, ssp, model, version=None, metadata=None, write_path=None):
    '''
    Helper function to cycle through range of years and generate gdp files. Writes gdp file to disk.

    Gnarly but necessary

    Parameters
    ----------

    nightlights_path: str
        path to nightlights gpd_ratio file

    baseline_gdp_path: str
        path to file with gdp baseline values

    growth_path: str
        path to file with gdp growth rates 

    ssp: str
        one of the following: 'SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5',

    econ_model: str
        one of the following: 'high', 'low'

    metadata: dict
        file specific metadata

    write_path: str
        file to write to

    Returns
    -------
    None 

    '''
        
    #format for writing baseline year
    base_write_path = write_path.format(ssp=ssp, model=model,version= version ,year=2010)

    #get your baseline gdp data
    base = gen_gdp_baseline(nightlights_path, baseline_gdp_path, ssp, model, base_year=2010, write_path=base_write_path)

    #create series of unique ISOs in gdp dataset
    iso = pd.Series(base.hierid.values).str.split('.').apply(lambda x: x[0])

    #align coords of is to hierid
    base.coords['iso'] = ('hierid', iso)

    #get your growth rate
    growth = get_growth_rates(growth_path)
    #get the base year gdp_growth
    growth_year = growth.sel(year=2010, model=model, scenario=ssp)

    # #growth_year = growth_year.fillna(growth_year.growth.sel(iso='mean'))
    # #growth_year = growth_year.loc[{'iso': growth.iso != 'mean'}]


    # ####################
    # # reindex for math #
    growth_year = growth_year.reindex_like(xr.Dataset(coords={'iso': np.unique(base.iso)}))
    growth_year = growth_year.sel(iso=base.iso)
    growth_year.coords['hierid'] = (('iso'),base.hierid)
    growth_year = growth_year.swap_dims({'iso': 'hierid'})

    #growth_year = reindex_growth_rate(growth, base, ssp, model, 2010)

    #do math
    annual= xr.Dataset()
    annual['gdppc'] = base['gdppc']*growth_year['growth']
    #metadata for base year
    metadata['year'] = str(2010)
  
    annual.attrs.update(metadata)
    annual = annual.drop('model').drop('scenario').drop('year')
    annual.to_netcdf(base_write_path)


    #calculate annual for each year
    for year in range(2011, 2100):
        if year %5 == 1:
            year = year - 1
            growth_year = growth.sel(year=year, model=model, scenario=ssp)
            growth_year = growth_year.reindex_like(xr.Dataset(coords={'iso': np.unique(base.iso)}))
            growth_year = growth_year.sel(iso=base.iso)
            growth_year.coords['hierid'] = (('iso'),base.hierid)
            growth_year = growth_year.swap_dims({'iso': 'hierid'})
            # growth_year = growth_year.drop('year')
            #growth_year = reindex_growth_rate(growth, base, ssp, model, year)


        annual['gdppc'] = annual['gdppc']*growth_year['growth']

        metadata['year'] = str(year)

        annual.attrs.update(metadata)
        annual = annual.drop('model').drop('scenario').drop('year')

        print(annual)
        
        if write_path:
            annual_write_path = write_path.format(ssp=ssp, model=model,version=version, year=year)
            if not os.path.isdir(os.path.dirname(annual_write_path)):
              os.makedirs(os.path.dirname(annual_write_path))

            annual.to_netcdf(annual_write_path)

def reindex_growth_rate(growth_ds, base, ssp, model, year):
    '''
    
    Performs Xarray Dataset manipulations to generate subsets of data from larger xarray dataset
    

    Parameters
    ----------
    growth_ds: 

    base: 

    ssp: str
        
    model: str

    year: str

    Returns
    -------
    Xarray Dataset
    '''
    growth_year = growth_ds.sel(year=year, model=model, scenario=ssp)
    growth_year = growth_year.reindex_like(xr.Dataset(coords={'iso': np.unique(base.iso)}))
    growth_year = growth_year.sel(iso=base.iso)
    growth_year.coords['hierid'] = (('iso'),base.hierid)
    growth_year = growth_year.swap_dims({'iso': 'hierid'})

    return growth_year


def gen_kernel_covars(covariate_paths, rcp, kernel=None):
    ''' 
    Computes kernelized covariate from a series of length of kernel

    Parameters
    ----------
    covariate_paths: list
      list of strings for file paths

    kernel: int
      duration to compute window

    climate: bool
      whether or not you are computing climate covariates

    Returns 
    -------
    xarray Dataset

    '''

    ####################################
    # Construct larger rolling dataset #
    ####################################

    datasets = None
    for y in range(1981, 2100):
      if y <= 2005:
            read_rcp = 'historical'
      else:
            read_rcp = rcp

      covar_path = covar_path_brc.format(rcp=read_rcp, model=model,variable=variable, year=y)
        

      with xr.open_dataset(covar_path) as ds:
        ds.load()

      ds = ds.mean(dim='time')
      ds.coords['year'] = y

      if datasets is None:
        values = [ds]
      else:
        values = [datasets, ds]

      datasets = xr.concat(values, dim='year')

      if y > 1982:
          valid_years = range(max(y-(kernel-1), 1981), y+1)
          datasets = datasets.sel(year=valid_years)
      
      ##################
      # Compute kernel #
      ##################    
      ds = gen_smoothed_covars(datasets, dim='year', kernel=kernel)

    return ds

    



def gen_smoothed_covars(ds, dim='year', kernel= None):
    '''
    Calculates the smooth average for a series

    Parameters
    ----------

    ds: Xarray Dataset

    dim: str
      dimension to smooth over

    kernel: int
      window to smooth over

    Returns
    -------
    Xarray Dataset

    Example
    -------

    .. code-block:: python


        In [1]: ds = xr.Dataset({'temperature': (['location', 'year'], 
           ...: np.random.randint(0,100,100).reshape(10,10))},
           ...: coords={'location': ['loc' + str(i) for i in range(10)], 
           ...: 'year' :[str(yr) for yr in range(2010, 2020)]})
        In [2]: ds
        Out[2]: 
        <xarray.Dataset>
        Dimensions:      (location: 10, year: 10)
        Coordinates:
          * location     (location) |S4 'loc0' 'loc1' 'loc2' 'loc3' 'loc4' 'loc5' ...
          * year         (year) |S4 '2010' '2011' '2012' '2013' '2014' '2015' '2016' ...
        Data variables:
            temperature  (location, year) int64 98 52 81 55 11 42 0 16 28 40 89 35 ...
        In [3]: gen_smoothed_covars(ds, dim='year', kernel=10)
        Out[3]:
        <xarray.Dataset>
        Dimensions:      (location: 10)
        Coordinates:
          * location     (location) |S4 'loc0' 'loc1' 'loc2' 'loc3' 'loc4' 'loc5' ...
        Data variables:
            temperature  (location) float64 56.78 60.82 33.91 56.29 59.33 60.4 48.11 ...

    ''' 
    ######################
    # smoothing function #
    ######################    
    print(ds)                                              
    smooth_array = triangle_smooth(kernel)

    ##############################################
    # handle sets of years less then kernel size #
    ##############################################
    if ds.dims[dim] < len(smooth_array):

        smooth_array = smooth_array[-ds.dims[dim]:]

    k = xr.DataArray(smooth_array, dims=(dim,), coords={dim: ds.coords[dim]})
    
    
    return (ds * k).sum(dim=dim)


def triangle_smooth(k):
  '''
  Produces the smoothed upper triangle array. 
  Because xarray Datasets are constructed with the larger year
  to the 'right' of the dataset we make sure the higher value 
  ratios are to the right to give more emphasis to recent values.

  Parameters
  ----------
  k: int

  Returns
  -------
  numpy array of length k

  Example
  -------

  .. code-block:: python

      In [1]: smooth = triangle_smooth(10)
      In [2[: smooth
      array([ 0.        ,  0.02222222,  0.04444444,  0.06666667,  0.08888889,
      0.11111111,  0.13333333,  0.15555556,  0.17777778,  0.2       ])
  '''

  smooth_array = np.arange(k)
  smooth_array = smooth_array/float(smooth_array.sum())

  return smooth_array
