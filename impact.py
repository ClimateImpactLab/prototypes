import xarray as xr
import pandas as pd
import numpy as np


class Impact():
  '''
    Base class for computing an impact as specified by the Climate Impact Lab

  '''
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
    self.weather = self.get_annual_weather(weather)

  def get_annual_weather(self, weather):
    '''
    Constructs the annual weather dataset for a given years impact

    Returns
    -------

      Dataset
        :py:class `~xarray Dataset` with each weather variable as an `~xarray DataArray`
    '''
    weather_files = [weather.format(pred=pred) for pred in self.preds]

    annual_weather = xr.Dataset()
    for file in weather_files:
      with xr.open_dataset(file) as ds:
          ds.load()
      varname = ds.variable
      annual_weather[varname] = ds[varname]


    return annual_weather

  def compute_betas(self, covars):
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

    betas = sum((self.gammas*covars).data_vars.values())

    return betas


  def compute(self,  
              gdp_covars,
              clim_covars,
              min_function=None,
              min_max=None,
              min_write_path=None,
              impact_function=None,
              postprocess_daily=False,
              postprocess_annual=False):
    '''
    Computes an impact for a unique set of gdp, climate, weather and gamma coefficient inputs.
    For each set of these, we take the analytic minimum value between two points and 
    

    Parameters
    ----------


    Returns
    -------
      :py:class `~xarray.Dataset` of impacts by hierid by outcome group 

    '''
    #Generate Betas
    betas = self.compute_betas(clim_covars, gdp_covars)

    #Compute Raw Impact
    impact= self._impact_function(betas, self.annual_weather)

    #Compute the min for flat curve adaptation
    if min_function:
      m_star = self._compute_m_star(betas, min_max, min_write_path)
      #Compare values and evaluate a max
      impact = np.minimum(impact, m_star)

    if postprocess_daily:
      impact = postprocess_daily(impact)
    if postprocess_annual:
      impact = postprocess_annual(impact)

    #Sum to annual, substract baseline, normalize 
    impact_annual = impact.sum(dim='time')  

    return impact_annual

  def _impact_function(self, betas, annual_weather):
    raise NotImplementedError

  def _compute_m_star(self, betas, min_max, min_write_path):
    raise NotImplementedError

  def postprocess_daily(self, impact):
    raise NotImplementedError

  def postprocess_annual(self, impact):
    raise NotImplementedError
