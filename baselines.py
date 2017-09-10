import xarray as xr



def impact_baseline(gammas, 
                      spec, 
                      weather, 
                      gdp_covar_path, 
                      climate_covar_path, 
                      metadata, 
                      base_years,
                      impact_function=None,
                      write_path=None):
    '''
    precomputes the baseline impact from beginning year to end year

    Parameters
    ----------
    weather_model_paths: str
        unformatted str variable for paths to annual weather

    gdp_covar_path: str
        baseline gdp path

    climate_covar_path: str
        baseline tavg climate path

    gammas: :py:class:`~xarray.Dataset` of gammas 

    metadata: dict

    begin: int 
        year to begin baseline calculation

    end: int
        year to end baseline calculation


    Returns
    -------

    Dataset
        returns a new `~xarray.Dataset` of baseline impacts

    '''

    if os.path.isfile(write_path):
        return self._get_baseline(write_path)


    #Construct Multi-year avg climate values
    base_weather= xr.Dataset()
    annual_weather_paths_tas = weather.format(scenario='{scenario}', 
                                                            year='{year}', 
                                                            model=metadata['model'], 
                                                            poly='')
    base_weather['tas'] = _construct_baseline_weather(annual_weather_paths_tas, metadata, begin, end)['tas']

    for p in range(2, poly + 1):
        annual_weather_paths_poly_tas = weather.format(scenario='{scenario}', 
                                                                    year='{year}', 
                                                                    model=metadata['model'], 
                                                                    poly='-poly-{}'.format(p))

        base_weather['tas-poly-{}'.format(p)] = _construct_baseline_weather(annual_weather_paths_poly_tas, metadata, begin, end)['tas-poly-{}'.format(p)]

    #Load Covars
    with xr.open_dataset(gdp_covar_path) as gdp_covar:
        gdp_covar.load()

    with xr.open_dataset(climate_covar_path) as clim_covar:
        clim_covar.load()

    #compute impacts
    base_impact = xr.Dataset()

    base_impact['baseline'] =  impact_function(base_weather, clim_covar, gdp_covar, gammas)

    #update metadata
    metadata['baseline_years'] = str([begin, end])
    metadata['dependencies'] = str([weather_model_paths, gdp_covar_path, climate_covar_path, gammas])
    metadata['oneline'] = 'Baseline impact value for mortality'
    metadata['description'] = 'Baseline impact value for mortality. Values are daily expected damage resolved to GCP hierid level region.'

    base_impact.attrs.update(metadata)

    if write_path:
        if not os.path.isdir(os.path.dirname(write_path)):
              os.makedirs(os.path.dirname(write_path))
        base_impact.to_netcdf(write_path)

    return base_impact

def _construct_baseline_weather(model_paths, metadata, begin, end):
    '''
    Constructs Tavg for baseline period for each of the climate variables

    Parameters
    ----------

    model_paths: str
        unformatted string for paths

    metadata: dict
        args for this spec including: ssp, model, econ_model, scenario, seed, etc

    begin: int

    end: int

    Returns
    -------

    `xarray.DataSet`
    '''
    years = []
    datasets = []
    for year in range(begin, end+1):
        if year <= 2005:
            read_rcp = 'historical'
        else: 
            read_rcp = metadata['rcp']

        path = model_paths.format(scenario=read_rcp ,year=year)
        with xr.open_dataset(path) as ds:
            ds.load()
        ds = ds.mean(dim='time')
        datasets.append(ds)
        years.append(year)

    ds_concat = xr.concat(datasets, pd.Index(years, name='year')) 
    ds_concat = ds_concat.mean(dim='year')
    return ds_concat

@memoize
def _get_baseline(base_path):
    '''
    Returns the cached version of the baseline impact

    '''
    with xr.open_dataset(base_path) as ds: 
      ds.load()

    return ds