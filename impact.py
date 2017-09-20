import os
import xarray as xr
import pandas as pd
import numpy as np
from toolz import memoize
import time


class Impact(object):
  '''
    Base class for computing an impact as specified by the Climate Impact Lab

  '''
  min_function = NotImplementedError

  def __init__(self, weather, preds, metadata):
    '''
    Parameters
    ----------
    weather: str
      String representation of the unformatted file paths for annual weather

    preds: list
      list of strings representing the values to complete formatting for annual weather

    metadata: dict
      metadata for this impact run

    Returns
    -------
    None
    '''
    self.preds = preds
    self.weather = self.get_weather(weather, metadata)

  def get_weather(self, weather, metadata):
    '''
    Constructs the annual weather dataarray for a given years impact

    Parameters
    ----------
    weather: str representation of unformatted file paths

    metadata: dict
      values to populate file string

    Returns
    -------

      Dataarray
        :py:class `~xarray Datarray` with each weather variable as an `coord`
    '''

    weather_files = [weather.format(scenario=metadata['scenario'], 
                                    model=metadata['model'], 
                                    year=metadata['year'], 
                                    pred=pred) for pred in self.preds]

    weathers = []
    for file in weather_files:
      with xr.open_dataset(file) as ds:
          ds.load()
      varname = ds.variable
      weathers.append(ds[varname])

    annual_weather = xr.concat(weathers, pd.Index(self.preds, name='prednames'))


    return annual_weather

 

  def compute_betas(self, gammas, gdp_covar, clim_covar):
    '''
    Computes the matrices beta*gamma x IR for each covariates 
    
    Parameters
    ----------
    gammas: :py:class `~xarray.Dataset`
        Coefficients for pred/covar combo

    gdp_covar: :py:class:`~xarray.DataArray`
      hierid by gdp dataarray

    clim_covar: :py:class:`~xarray.DataArray`
      hierid by clim dataarray    
 
    Returns
    -------
      Dataarray
        hierid by outcome and predname :py:class `~xarray.Dataarray` 
    '''
    
    betas = (gammas*covars).sum(dim='covarnames')

    return betas

  def compute(self,  
              gammas, 
              gdp_covar,
              clim_covar,
              baseline,
              bounds=None,
              t_star_path=None,
              postprocess_daily=False,
              postprocess_annual=False):
    '''
    Computes an impact for a unique set of gdp, climate, weather and gamma coefficient inputs.
    For each set of these, we take the analytic minimum value between two points, 
    save t_star to disk and compute analytical min for function m_star for a givene covariate set
    This operation is called for every adaptation scenario specified in the run script
    
    Parameters
    ----------

    gammas: 
      covarname by outcome py:class: `~xarray.Dataset`

    gdp_covar: :py:class:`~xarray.DataArray`
      hierid by gdp dataarray

    clim_covar: :py:class:`~xarray.DataArray`
      hierid by clim dataarray 

    baseline: :py:class:`~xarray.Dataset`
      precomputed avg baseline impact for impacts between base years

    bounds: list
      list of values to compute mins for computing m-star

    t_star_path: str
      unformatted string path to designate read/write location for t_star
    
    postprocess_daily: bool
      If true execute additional functions in subclasses
  
    postprocess_annual: bool
      If true execute additional functions in subclasses


    Returns
    -------
      :py:class `~xarray.Dataset` of impacts by hierid by outcome group 

    '''
    t1 = time.time()
    #Generate Betas
    betas = self.compute_betas(gammas, gdp_covar, clim_covar)
    t2 = time.time()
    print('computing betas {}'.format(t2-t1))

    #Compute Raw Impact
    impact = self.impact_function(betas, self.weather)
    t3 = time.time()
    print('computing impact {}'.format(t3-t2))

    #Compute the min for flat curve adaptation
    m_star = self._compute_m_star(betas, bounds=bounds, t_star_path=t_star_path)
    t4 = time.time()
    print('computing m_star {}'.format(t4-t3))

    #Compare values and evaluate a max
    impact = xr.ufuncs.minimum(impact, m_star)
    t5 =time.time()
    print('taking min {}'.format(t5-t4))

    if postprocess_daily:
      impact = self.postprocess_daily(impact)

    #Sum to annual
    impact = impact.sum(dim='time')
    t6 = time.time() 
    print('annual sum {}'.format(t6 -t5))

    impact_annual = impact - baseline
    t7 = time.time()
    print('rebased {}'.format(t7 -t6))


    if postprocess_annual:
      impact_annual= self.postprocess_annual(impact_annual) 

    impact_annual = impact_annual.rename({'__xarray_dataarray_variable__': 'rebased'})

    return impact_annual.rebased

  @memoize
  def _get_t_star(self, path):
    '''
    Read precomputed t_star

    Parameters
    ----------
    path: str
      place to load t-star from 

    '''
    with xr.open_dataset(path) as ds:
      ds.load()
    return ds

  def _compute_m_star(self, betas, bounds=None, t_star_path=None):
    '''
    Computes m_star, the value of an impact function for a given set of betas given t_star. 
    t_star, the value t at which an impact is minimized for a given hierid is precomputed 
    and used to compute m_star.

    Parameters
    ----------
    betas: :py:class `~xarray.Dataset` 
        coefficients by hierid Dataset 

    bounds: list
        values to evaluate min at 

    t_star_path: str

    Returns
    -------

    m_star Dataset
        :py:class`~xarray.Dataset` of impacts evaluated at tstar. 


    .. note:: writes to disk and subsequent calls will read from disk. 
    '''
    # if file does not exist create it
    if not os.path.isfile(t_star_path):

        #Compute t_star according to min function
        t_star = self.compute_t_star(betas, bounds=bounds)
        #write to disk
        if not os.path.isdir(os.path.dirname(t_star_path)):
                os.makedirs(os.path.dirname(t_star_path))

        t_star.to_netcdf(t_star_path)

    else:

        #Read from disk
        t_star = self._get_t_star(t_star_path)

    return self.impact_function(betas, t_star)

  def compute_t_star(self, betas, bounds=None):
    return self.min_function(betas, bounds=bounds)

  def impact_function(self, betas, annual_weather):
    raise NotImplementedError

  def postprocess_daily(self, impact):
    raise NotImplementedError

  def postprocess_annual(self, impact):
    raise NotImplementedError

  
