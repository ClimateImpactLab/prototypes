import os
import xarray as xr
import pandas as pd
import numpy as np
from toolz import memoize
from mins import minimize_polynomial
import time


def construct_baseline_weather(pred, pred_path, metadata, base_years):
    '''
    Constructs Tavg for baseline period for each of the climate variables

    Parameters
    ----------

    pred: str
        key for pred values

    pred_paths: str
        unformatted string for paths

    metadata: dict
        args for this spec including: ssp, model, econ_model, scenario, seed, etc

    base_years: list
        list of ints of base years

    Returns
    -------
        DataArray
        :py:class:`~xarray.DataArray` of predname by hierid
    '''
    years = []
    das = []
    for year in range(base_years[0], base_years[1]+1):
        if year <= 2005:
            read_rcp = 'historical'
        else: 
            read_rcp = metadata.get('scenario', 'rcp85')

        path = pred_path.format(scenario=read_rcp ,year=year, model=metadata['model'])
        with xr.open_dataset(path) as ds:
            da = ds[pred].load()
        das.append(da.mean(dim='time'))
        years.append(year)

    das_concat = xr.concat(das, pd.Index(years, name='year')) 
    
    return das_concat.mean(dim='year')

def construct_weather(weather, baseline=True, metadata=None, base_years=None):
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
        print(baseline)
        if baseline:
            weather_data.append(construct_baseline_weather(pred, path, metadata, base_years))
            print(weather_data)
        else:
            with xr.open_dataset(path.format(**metadata)) as ds:
                weather_data.append(ds[pred].load())
        prednames.append(pred)

    return xr.concat(weather_data, pd.Index(prednames, name='prednames'))

def construct_covars(add_constant=True, **covars):
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

    covarnames = covars.keys()
    covar_data = []
    for covar in covarnames:
        with xr.open_dataset(covars[covar]) as ds:
            covar_data.append(ds[covar].load())

    if add_constant:
        ones = xr.DataArray(np.ones(shape=covar_data[0].shape),
            coords=covar_data[0].coords,
            dims=covar_data[0].dims)
        covarnames.append('1')
        covar_data.append(ones)
        
    return xr.concat(covar_data, pd.Index(covarnames, name='covarnames'))


def basline_to_netcdf(betas, base_years, metadata, write_path):
    '''
    Helper function to update metadata and write baseline to disk

    baseline: DataArray
        :py:class:`~xarray.DataArray` of hierid by predname by outcome

    base_years: list
        begin and end 

    metadata: dict
        values to populate Dataset metadata attrs

    write_path: str
        place to save precomputed dataset



    Returns
    -------
    None

    '''

    baseline_ds= xr.Dataset()
    baseline_ds['baseline'] = baseline

    metadata['baseline_years'] = str(base_years)
    metadata['oneline'] = 'Baseline impact value for {variable}'.format(variable= metadata['variable'])
    metadata['description'] = 'Baseline impact value for {variable}. Values are annual expected damage resolved to GCP hierid level region.'.format(variable=metadata['variable'])

    varattrs = {k:str(v) for k,v in metadata.items()}
    baseline_ds.attrs.update(varattrs)

    if not os.path.isdir(os.path.dirname(write_path)):
          os.makedirs(os.path.dirname(write_path))
    
    baseline_ds.to_netcdf(write_path)
        

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
            gammas, 
            covars,
            baseline=None,
            bounds=None,
            clip_flat_curve=True,
            t_star_path=None):
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

        #Generate Betas
        betas = (gammas*covars).sum(dim='covarnames')

        #Compute Raw Impact
        impact = self.impact_function(betas, weather)

        if clip_flat_curve:

            t_star = self.get_t_star(betas, bounds, t_star_path)
            #Compute the min for flat curve adaptation
            impact_flatcurve = self.impact_function(betas, t_star)

            #Compare values and evaluate a max
            impact = xr.ufuncs.minimum(impact, impact_flatcurve)

        impact = self.postprocess_daily(impact)

        if baseline is None:
            return impact

        #Sum to annual
        impact = impact.sum(dim='time')

        impact_annual = impact - baseline

        impact_annual = self.postprocess_annual(impact_annual) 

        return impact_annual

    def get_t_star(self, betas, bounds, path):
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
