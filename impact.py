import os
import xarray as xr
import pandas as pd
import numpy as np
from toolz import memoize


class Impact(object):
  '''
    Base class for computing an impact as specified by the Climate Impact Lab

  '''

  min_function = NotImplemented

  def __init__(self, weather, preds):
    '''
    Parameters
    ----------
    weather: str
      String representation of the unformatted file paths for annual weather

    preds: list
      list of strings representing the values to complete formatting for annual weather

    '''
    self.preds = preds
    self.weather = self._get_annual_weather(weather)

  def _get_annual_weather(self, weather):
    '''
    Constructs the annual weather dataset for a given years impact

    Returns
    -------

      Dataset
        :py:class `~xarray Dataset` with each weather variable as an `~xarray DataArray`
    '''
    weather_files = [weather.format(pred=pred) for pred in self.preds]

    weathers = []
    for file in weather_files:
      with xr.open_dataset(file) as ds:
          ds.load()
      varname = ds.variable
      weathers.append(ds[varname])

    annual_weather = xr.concat(weathers, pd.Index(self.preds, name='prednames'))


    return annual_weather

  def _compute_betas(self, gammas, covars):
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
    betas = xr.Dataset()
    
    covars = xr.merge(covars)

    #add intercept for easy math
    covars['1'] = ('hierid', ), np.ones(len(covars.hierid))

    betas = sum((gammas*covars).data_vars.values())

    return betas

  def compute(self,  
              gammas, 
              gdp_covars,
              clim_covars,
              min_max_boundary=None,
              t_star_write_path=None,
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
    #Generate Betas
    betas = self._compute_betas(gammas, [clim_covars,gdp_covars])

    #Compute Raw Impact
    impact = self.impact_function(betas, self.weather)

    #Compute the min for flat curve adaptation
    m_star = self.compute_m_star(betas, min_max_boundary, t_star_write_path)

    #Compare values and evaluate a max
    impact = xr.ufuncs.minimum(impact, m_star)

    if postprocess_daily:
      impact = self.postprocess_daily(impact)

    #Sum to annual
    impact_annual = impact.sum(dim='time') 

    if postprocess_annual:
      impact_annual = self.postprocess_annual(impact_annual) 

    return impact_annual

  @memoize
  def _get_t_star(self, path):
    with xr.open_dataset(path) as ds:
      ds.load()
    return ds

  def compute_m_star(self, betas, min_max_boundary=None, t_star_write_path=None):
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
    if not os.path.isfile(t_star_write_path):

        #Compute t_star according to min function
        t_star = self.min_function(betas, min_max_boundary)
        #write to disk
        if not os.path.isdir(os.path.dirname(t_star_write_path)):
                os.path.makedir(os.path.dirname(t_star_write_path))

        t_star.to_netcdf(t_star_write_path)

    #Read from disk
    t_star = self._get_t_star(t_star_write_path)
      
    return sum((t_star*betas).data_vars.values()).sum(dim='prednames')

  def impact_function(self, betas, annual_weather):
    raise NotImplementedError

  def postprocess_daily(self, impact):
    raise NotImplementedError

  def postprocess_annual(self, impact):
    raise NotImplementedError

  
