

class Impact(gammas, weather, covariates):
  '''
    Base class for computing an impact as specified by the Climate Impact Lab

  '''
  def __init__(self):
    self.weather = self.get_annual_weather(weather, spec)
    self.betas = self.compute_betas(gammas, covariates)
    self.impact = self.compute_impact(self.weather, self.betas)
    self.gammas = gammas


  def precompute_baseline(weather_model_paths, 
                        gdp_covar_path, 
                        climate_covar_path, 
                        gammas, 
                        metadata, 
                        begin, 
                        end, poly=None, write_path=None):
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
        return get_baseline(write_path)


    #Construct Multi-year avg climate values
    base = xr.Dataset()
    annual_weather_paths_tas = weather_model_paths.format(scenario='{scenario}', 
                                                            year='{year}', 
                                                            model=metadata['model'], 
                                                            poly='')
    base['tas'] = _construct_baseline_weather(annual_weather_paths_tas, metadata, begin, end)['tas']

    for p in range(2, poly + 1):
        annual_weather_paths_poly_tas = weather_model_paths.format(scenario='{scenario}', 
                                                                    year='{year}', 
                                                                    model=metadata['model'], 
                                                                    poly='-poly-{}'.format(p))
        base['tas-poly-{}'.format(p)] = _construct_baseline_weather(annual_weather_paths_poly_tas, metadata, begin, end)['tas-poly-{}'.format(p)]

    #Load Covars
    with xr.open_dataset(gdp_covar_path) as gdp_covar:
        gdp_covar.load()

    with xr.open_dataset(climate_covar_path) as clim_covar:
        clim_covar.load()

    #compute impacts
    base_impact = xr.Dataset()

    base_impact['baseline'] =  compute_polynomial(base, clim_covar, gdp_covar, gammas)

    #update metadata
    metadata['baseline_years'] = str([begin, end])
    metadata['dependencies'] = str([weather_model_paths, gdp_covar_path, climate_covar_path, gammas])
    metadata['oneline'] = 'Baseline impact value for mortality'
    metadata['description'] = 'Baseline impact value for mortality. Values are annual/daily expected damage resolved to GCP hierid/country level region.'

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


  def get_annual_weather(weather, spec):
    '''
    Constructs the annual weather dataset for a given years impact

    Parameters
    ----------
    models_paths: list
        unformatted string representing path to weather variables for that year

    spec: dict
      describes how to format input file paths for weather files

    Returns
    -------

      Dataset
        :py:class `~xarray Dataset` with each weather variable as an `~xarray DataArray`
    '''

    dataset = xr.Dataset()

    with xr.open_dataset(model_paths.format(poly='')) as ds:
        ds.load()

    varname = ds.variable
    dataset[varname] = ds[varname]
    

    for poly in range(2, polymomial+1):
        fp = model_paths.format(poly='-poly-{}'.format(poly))

        with xr.open_dataset(fp) as ds:
            ds.load()

        varname = ds.variable
        dataset[varname] = ds[varname]
    return dataset

    return weather

  def compute_betas(gammas, covariates):
    '''
    Computes the matrices beta*gamma x IR for each covariates 

    1. Calls method to get gammas at given p-value
    2. Calls method toompute gdp covariate
    3. Calls method to compute tas covariate
    4. Computes outer product of 

    Parameters
    ----------
    clim_path: str
        path to climate covariate

    gdp_path: str
        path to gdp covariate

    ssp: str
        one of the following: 'SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5',

    econ_model: str
        one of the following: 'high', 'low'
 
    Returns
    -------
    Xarray Dataset with values for each of the Betas

    '''
    betas = xr.Dataset()

    betas['tas'] = (gammas['beta0_pow1'] + gammas['gdp_pow1'] * gdp_covar['gdppc'] + gammas['tavg_pow1']*clim_covar['tas'])
    betas['tas-poly-2'] = (gammas['beta0_pow2'] + gammas['gdp_pow2'] * gdp_covar['gdppc'] + gammas['tavg_pow2']*clim_covar['tas'])
    betas['tas-poly-3'] = (gammas['beta0_pow3'] + gammas['gdp_pow3'] * gdp_covar['gdppc'] + gammas['tavg_pow3']*clim_covar['tas'])
    betas['tas-poly-4'] = (gammas['beta0_pow4'] + gammas['gdp_pow4'] * gdp_covar['gdppc'] + gammas['tavg_pow4']*clim_covar['tas'])

    return betas


  def _functioal_form(betas, weather):
    raise NotImplementedError('You need to specify a functional form')


  def compute_impact(betas, weather, impact=None):

      baseline = self.compute_baseline()




    return impact

  def postprocess_daily(impact):
    return

  def postprocess_annual(impact):
    return
