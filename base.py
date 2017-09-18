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
        with xr.open_dataset(base_path) as ds: 
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

        `xarray.DataSet`
        '''
        years = []
        datasets = []
        for year in range(base_years[0], base_years[1]+1):
            if year <= 2005:
                read_rcp = 'historical'
            else: 
                read_rcp = 'rcp85'

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


        base_weather = xr.concat(ar, pd.Index(preds, name='predname'))

        return base_weather




    def compute(self, gammas, gdp_covars, clim_covars):

        if os.path.isfile(self.base_path):
            return self._get_baseline(self.base_path)

        if not self.weather_computed:
            self.weather_computed = self.get_weather(self.weather_paths, self.preds, self.metadata)


        betas = self._compute_betas(gammas, [clim_covars, gdp_covars])

        #Compute Raw Impact
        impact = self.impact_function(betas, self.weather_computed)

        return impact



    def impact_function(self, betas, weather):


        impact = (betas*weather).sum(dim='prednames')

        self._basline_to_netcdf(impact, self.metadata, self.base_path)


        return impact


    def _basline_to_netcdf(self, impact_da, metadata, write_path):
        '''
        Updates metadata and writes baseline to disk

        '''
        impact_da.rename({'__xarray_dataarray_variable__': 'baseline'})


        metadata['baseline_years'] = str(self.base_years)
        metadata['oneline'] = 'Baseline impact value for mortality'
        metadata['description'] = 'Baseline impact value for mortality. Values are annual expected damage resolved to GCP hierid level region.'

        impact_da.attrs.update(metadata)


        if not os.path.isdir(os.path.dirname(write_path)):
              os.makedirs(os.path.dirname(write_path))
        
        impact_da.to_netcdf(write_path)

    ##############
    # Deprecated #
    ##############
    # def impact_baseline(gammas,
    #                       weather, 
    #                       gdp_covar_path, 
    #                       climate_covar_path, 
    #                       metadata, 
    #                       base_years,
    #                       impact_function=None,
    #                       write_path=None):
    #     '''
    #     precomputes the baseline impact from beginning year to end year

    #     Parameters
    #     ----------
    #     weather_model_paths: str
    #         unformatted str variable for paths to annual weather

    #     gdp_covar_path: str
    #         baseline gdp path

    #     climate_covar_path: str
    #         baseline tavg climate path

    #     gammas: :py:class:`~xarray.Dataset` of gammas 

    #     metadata: dict

    #     begin: int 
    #         year to begin baseline calculation

    #     end: int
    #         year to end baseline calculation


    #     Returns
    #     -------

    #     Dataset
    #         returns a new `~xarray.Dataset` of baseline impacts

    #     '''
    #     preds = gammas.prednames.values

    #     if os.path.isfile(write_path):
    #         return self._get_baseline(write_path)


    #     #Construct Multi-year avg climate values
    #     base_weather= xr.Dataset()
    #     annual_weather_paths_tas = weather.format(scenario='{scenario}', 
    #                                                             year='{year}', 
    #                                                             model=metadata['model'], 
    #                                                             poly='')
    #     base_weather['tas'] = _construct_baseline_weather(annual_weather_paths_tas, metadata, begin, end)['tas']

    #     for p in range(2, poly + 1):
    #         annual_weather_paths_poly_tas = weather.format(scenario='{scenario}', 
    #                                                                     year='{year}', 
    #                                                                     model=metadata['model'], 
    #                                                                     poly='-poly-{}'.format(p))

    #         base_weather['tas-poly-{}'.format(p)] = _construct_baseline_weather(annual_weather_paths_poly_tas, metadata, begin, end)['tas-poly-{}'.format(p)]

    #     #Load Covars
    #     with xr.open_dataset(gdp_covar_path) as gdp_covar:
    #         gdp_covar.load()

    #     with xr.open_dataset(climate_covar_path) as clim_covar:
    #         clim_covar.load()

    #     #compute impacts
    #     base_impact = xr.Dataset()

    #     base_impact['baseline'] =  impact_function(base_weather, clim_covar, gdp_covar, gammas)

    #     #update metadata
    #     metadata['baseline_years'] = str([begin, end])
    #     metadata['oneline'] = 'Baseline impact '
    #     metadata['description'] = 'Baseline impact value. Values are annual expected damage resolved to GCP hierid level region.'

    #     base_impact.attrs.update(metadata)

    #     if write_path:
    #         if not os.path.isdir(os.path.dirname(write_path)):
    #               os.makedirs(os.path.dirname(write_path))
    #         base_impact.to_netcdf(write_path)

    #     return base_impact



