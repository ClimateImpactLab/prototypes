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

    '''
    self.preds = preds
    self.weather = self.get_weather(weather, metadata)

  def get_weather(self, weather, metadata):
    '''
    Constructs the annual weather dataset for a given years impact

    Returns
    -------

      Dataset
        :py:class `~xarray Dataset` with each weather variable as an `~xarray DataArray`
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

  def _compute_betas(self, gammas, gdp_covar, clim_covar):
    '''
    Computes the matrices beta*gamma x IR for each covariates 
    
    Parameters
    ----------
    gammas: :py:class `~xarray.Dataset`
        Coefficients for pred/covar combo

    spec: 
      specifies the shape/structure of computation

    covars: list of :py:class `~xarray.Dataset`
        covariates for each hierid
    
 
    Returns
    -------
      :py:class `~xarray.Dataset` values for each predname beta

    '''
    
    gdp_covar = gdp_covar.drop('iso')
    ones = xr.DataArray(np.ones(len(gdp_covar.hierid)), coords={'hierid': gdp_covar.hierid}, dims=['hierid'], name='1')
    covars = xr.concat([ones, clim_covar, gdp_covar], pd.Index(gammas.data_vars.keys(), name='covarnames'))


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
    For each set of these, we take the analytic minimum value between two points and 
    

    Parameters
    ----------

    gammas: py:class: `~xarray.Dataset`


    Returns
    -------
      :py:class `~xarray.Dataset` of impacts by hierid by outcome group 

    '''
    t1 = time.time()
    #Generate Betas
    betas = self._compute_betas(gammas, gdp_covar, clim_covar)
    t2 = time.time()
    print(betas)

    print('computing betas {}'.format(t2-t1))

    #Compute Raw Impact
    impact = self.impact_function(betas, self.weather)
    t3 = time.time()
    print('computing impact {}'.format(t3-t2))
    print(impact)

    #Compute the min for flat curve adaptation
    m_star = self._compute_m_star(betas, bounds=bounds, t_star_path=t_star_path)
    print(m_star)
    t4 = time.time()
    print('computing m_star {}'.format(t4-t3))

    #Compare values and evaluate a max
    impact = xr.ufuncs.minimum(impact, m_star)
    print(impact)
    t5 =time.time()
    print('taking min {}'.format(t5-t4))

    if postprocess_daily:
      impact = self.postprocess_daily(impact)

    #Sum to annual
    impact = impact.sum(dim='time')
    t6 = time.time() 
    print('annual sum {}'.format(t6 -t5))



    print('impact', impact)
    print('baseline', baseline)
    impact_annual = impact - baseline
    print('rebased', impact_annual)
    t7 = time.time()
    print('rebased {}'.format(t7 -t6))


    if postprocess_annual:
      impact_annual= self.postprocess_annual(impact_annual) 


    return impact_annual

  @memoize
  def _get_t_star(self, path):
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

    min_max_boundary: np.array
        values to evaluate min at 

    min_function: minimizing function to compute tstar

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

  
