import os
import xarray as xr
import numpy as np
import pandas as pd
from toolz import memoize
from impact import Impact



class BaseImpact(Impact):
    '''
    
    '''

    def __init__(self, weather_paths, preds, metadata, base_years, base_path):
        self.weather_paths = weather_paths
        self.preds = preds
        self.metadata  = metadata
        self.base_years = base_years
        self.base_path = base_path
        self.weather_computed = None



    @memoize
    def _get_baseline(self, base_path):
        '''
        Returns the cached version of the baseline impact

        '''
        with xr.open_dataarray(base_path) as ds: 
          ds.load()

        return ds

    def _construct_baseline_weather(self, model_paths, metadata, base_years):
        '''
        Constructs Tavg for baseline period for each of the climate variables

        Parameters
        ----------

        model_paths: str
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
        datasets = []
        for year in range(base_years[0], base_years[1]+1):
            if year <= 2005:
                read_rcp = 'historical'
            else: 
                read_rcp = metadata.get('scenario', 'rcp85')

            path = model_paths.format(scenario=read_rcp ,year=year)
            with xr.open_dataset(path) as ds:
                ds.load()
            ds = ds.mean(dim='time')
            datasets.append(ds)
            years.append(year)

        ds_concat = xr.concat(datasets, pd.Index(years, name='year')) 
        ds_concat = ds_concat.mean(dim='year')
        return ds_concat


    def get_weather(self, weather, preds, metadata):
        '''
        Constructs the baseline weather by predname for computing baseline impact during base period

        .. note: overrides base Impact class `get_weather` method

        Parameters
        ----------
        weather: str
            unformatted path str for weather for baseline period scenario

        preds: list
            names of preds to build weather from

        metadata: dict
            used in constructing paths for weather io

        Returns
        -------
         DataArray
            :py:class:`~xarray.DataArray` of predname by hierid
         
        .. note:: overrides base Impact class `get_weather` method

        '''

        base_weather_pred = []
        
        for pred in preds:
            annual_weather_paths = weather.format(scenario='{scenario}', 
                                                            year='{year}', 
                                                            model=self.metadata['model'], 
                                                            pred=pred)

            base_weather_pred.append(self._construct_baseline_weather(annual_weather_paths, self.metadata, self.base_years))

        ar = []
        for i, pred in enumerate(preds):
            ar.append(base_weather_pred[i][pred])


        base_weather = xr.concat(ar, pd.Index(preds, name='prednames'))

        return base_weather




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
                self.weather_paths, self.preds, self.metadata)


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


