import os
import xarray as xr
import numpy as np
import pandas as pd
from toolz import memoize
from impact import Impact, construct_covars, construct_weather


def _construct_baseline_weather(pred, pred_path, metadata, base_years):
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
        da = da.mean(dim='time')
        das.append(da)
        years.append(year)

    das_concat = xr.concat(das, pd.Index(years, name='year')) 
    
    return das_concat.mean(dim='year')

def construct_weather(weather, metadata, base_years):
    '''
    Constructs the baseline weather by predname for computing baseline impact during base period

    .. note: overrides base Impact class `get_weather` method

    Parameters
    ----------
    weather: dict
        preds and unformatted path str for weather for baseline period scenario

    metadata: dict
        used in constructing paths for weather io

    base_years: list
        list of ints indicating begin and end of period

    Returns
    -------
     DataArray
        :py:class:`~xarray.DataArray` of predname by hierid
     
    '''

    base_weather_pred = []
    preds = []
    for pred,pred_path in weather.items():
        base_weather_pred.append(_construct_baseline_weather(pred, pred_path, metadata, base_years))
        preds.append(pred)
    
    return xr.concat(base_weather_pred, pd.Index(preds, name='prednames'))


class BaseImpact(Impact):
    '''
    
    '''
    @memoize
    def _get_baseline(self, base_path):
        '''
        Returns the cached version of the baseline impact

        '''
        with xr.open_dataarray(base_path) as ds: 
          ds.load()

        return ds




    def compute(self, gammas, gdp_covars, clim_covars):
        '''
        Computes the baseline impact for a given hierid over the baseline period

        Parameters
        ----------
        gammas: Dataset
            :py:class:`~xarray.Dataset` of covarname by outcome by predname

        gdp_covar: Dataarray
            :py:class:`~xarray.Dataarray` of gdp covariates by 


        clim_covar: Dataarray
            :py:class:`~xarray.Dataarray` of clim covariates by hierid


        Returns
        -------
        Dataset
            :py:class:`~xarray.Dataset` of impact by hierid by outcome for baseline period

        .. note:: overrides base Impact class `compute` method
    
        '''
        if os.path.isfile(self.base_path):
            return self._get_baseline(self.base_path)

        if self.weather_computed is None:
            self.weather_computed = self.get_weather(
                self.weather_paths, self.metadata)


        betas = self.compute_betas(gammas, gdp_covars, clim_covars)

        #Compute Raw Impact
        impact= self.impact_function(betas, self.weather_computed)

        return impact 


    def impact_function(self, betas, weather):
        '''
        Computes the dot product of betas and annual weather by outcome group.

        Writes dataset to disk for access by other years in the model spec run 

        Parameters
        ----------

        betas: DataArray
            :py:class:`~xarray.DataArray` of hierid by predname by outcome

        weather: DataArray
            :py:class:`~xarray.DataArray` of hierid by predname by outcome

        Returns
        -------
        Dataset
            :py:class:`~xarray.Dataset` of impact by hierid by outcome group for baseline period
        
        .. note:: overrides base Impact class `impact_function` method

        '''

        impact = (betas*weather).sum(dim='prednames')

        self._basline_to_netcdf(impact, self.metadata, self.base_path)

        impact_read = xr.open_dataarray(self.base_path)

        return impact_read


    def _basline_to_netcdf(self, betas, metadata, write_path):
        '''
        Helper function to update metadata and write baseline to disk

        betas: DataArray
            :py:class:`~xarray.DataArray` of hierid by predname by outcome

        metadata: dict
            values to populate Dataset metadata attrs

        write_path: str
            place to save precomputed dataset

        Returns
        -------
        None

        '''

        baseline = xr.Dataset()
        baseline['baseline'] = betas

        metadata['baseline_years'] = str(self.base_years)
        metadata['oneline'] = 'Baseline impact value for mortality'
        metadata['description'] = 'Baseline impact value for mortality. Values are annual expected damage resolved to GCP hierid level region.'

        varattrs = {k:str(v) for k,v in metadata.items()}
        baseline.attrs.update(varattrs)

        if not os.path.isdir(os.path.dirname(write_path)):
              os.makedirs(os.path.dirname(write_path))
        
        baseline.to_netcdf(write_path)


