import os
import xarray as xr
import pandas as pd
import numpy as np
from toolz import memoize
from mins import minimize_polynomial
import time


def construct_weather(weather, metadata):
    '''
    Helper function to build out weather dataarray

    Parameters
    ----------

    weather: dict
        dictionary of prednames and file paths for each predname

    Returns
    -------
    combined: DataArray
            Combined :py:class:`~xarray.DataArray` of weather
            variables, with variables concatenated along the
            new `prednames` dimension


    '''
    prednames = []
    weather_data = []
    for pred, path in weather.items():
        with xr.open_dataset(path.format(pred=pred, **metadata)) as ds:
            weather_data.append(ds[pred].load())
        prednames.append(pred)

    return xr.concat(weather_data, pd.Index(prednames, name='prednames'))

def construct_covars(covars, add_constant=True):
    '''
    Helper function to construct the covariates dataarray

    Parameters
    -----------
    add_constant : bool
        flag indicating whether a constant term should be added. The constant term will have the
        same shape as the other covariate DataArrays
    
    covars: keyword arguments of DataArrays
      covariate :py:class:`~xarray.DataArray`s

    Returns
    -------
    combined: DataArray
        Combined :py:class:`~xarray.DataArray` of covariate
        variables, with variables concatenated along the
        new `covarnames` dimension
    '''
    covarnames = []
    covar_data = []
    for covar, path  in covars.items():
        with xr.open_dataset(path) as ds:
            covar_data.append(ds[covar].load())
            covarnames.append(covar)

    if add_constant:
        ones = xr.DataArray(np.ones(shape=covar_data[0].shape),
            coords=covar_data[0].coords,
            dims=covar_data[0].dims)
        covarnames.append('1')
        covar_data.append(ones)
        
    return xr.concat(covar_data, pd.Index(covarnames, name='covarnames'))
        

class Impact(object):
    '''
    Base class for computing an impact as specified by the Climate Impact Lab
    
    '''

    min_function = NotImplementedError


    def impact_function(self, betas, weather):
        '''
        computes the dot product of betas and annual weather by outcome group

        Parameters
        ----------

        betas: DataArray
            :py:class:`~xarray.DataArray` of hierid by predname by outcome

        weather: DataArray
            :py:class:`~xarray.DataArray` of hierid by predname by outcome

        Returns
        -------
        DataArray
            :py:class:`~xarray.DataArray` of impact by outcome by hierid

        .. note::

            overrides `impact_function` method in Impact base class

        '''
    
        return (betas*weather).sum(dim='prednames')
    

    def compute(self,
            weather,
            betas,
            clip_flat_curve=True,
            t_star=None):
        '''
        Computes an impact for a unique set of gdp, climate, weather and gamma coefficient inputs.
        For each set of these, we take the analytic minimum value between two points, 
        save t_star to disk and compute analytical min for function m_star for a givene covariate set
        This operation is called for every adaptation scenario specified in the run script

        Parameters
        ----------
        
        weather: DataArray
          weather :py:class:`~xarray.DataArray`

        gammas: DataArray
          covarname by outcome py:class:`~xarray.DataArray`

        gdp_covar: DataArray
          hierid by gdp :py:class:`~xarray.DataArray`

        clim_covar: DataArray
          hierid by predname :py:class:`~xarray.DataArray`

        baseline: :py:class:`~xarray.Dataset`
          precomputed avg baseline impact for impacts between base years

        bounds: list
          list of values to compute mins for computing m-star
         
        clip_flat_curve: bool
          flag indicating that flat-curve clipping should be performed
          on the result

        t_star_path: str
          unformatted string path to designate read/write location for t_star


        Returns
        -------
          :py:class `~xarray.Dataset` of impacts by hierid by outcome group 

        '''

        #Compute Raw Impact
        impact = self.impact_function(betas, weather)

        if clip_flat_curve:

            #Compute the min for flat curve adaptation
            impact_flatcurve = self.impact_function(betas, t_star)

            #Compare values and evaluate a max
            impact = xr.ufuncs.minimum(impact, impact_flatcurve)

        impact = self.postprocess_daily(impact)

        #Sum to annual
        impact = impact.sum(dim='time')

        impact_annual = self.postprocess_annual(impact_annual) 

        return impact_annual

    def get_t_star(self,betas, bounds, path=None):
        '''
        Read precomputed t_star

        Parameters
        ----------

        betas: DataArray
            :py:class:`~xarray.DataArray` of betas as prednames by hierid

        bounds: list
            values between which to evaluate function

        path: str
          place to load t-star from 

        '''
        
        try:
            with xr.open_dataset(path) as t_star:
                return t_star.load()

        except OSError:
            pass

        except (IOError, ValueError):
            try:
                os.remove(path)
            except:
                pass

        #Compute t_star according to min function
        t_star = self.compute_t_star(betas, bounds=bounds)

        #write to disk
        if path != None:
            if not os.path.isdir(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            t_star.to_netcdf(path)

        return t_star

    def compute_t_star(self, betas, bounds=None):
        return self.min_function(betas, bounds=bounds)

    def postprocess_daily(self, impact):
        return impact

    def postprocess_annual(self, impact):
        return impact


class PolynomialImpact(Impact):
    '''
    Polynomial-specific Impact spec, with ln(gdppc) and climtas for covariates
    '''

    @staticmethod
    def min_function(*args, **kwargs):
        '''
        helper function to call minimization function for given mortality polynomial spec
        mortality_polynomial implements findpolymin through `np.apply_along_axis`

        Parameters
        ----------

        betas: DataArray
            :py:class:`~xarray.DataArray` of hierid by predname by outcome

        dim: str
            dimension to apply minimization to

        bounds: list
            values between which to search for t_star

        Returns
        -------
            :py:class:`~xarray.DataArray` of hierid by predname by outcome

        .. note:: overides `min_function` in Impact base class
        '''

        return minimize_polynomial(*args, **kwargs)
