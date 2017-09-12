import xarray as xr
import numpy as np
import toolz



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


def _compute_tstar(coeffs, min_max):
    '''
    Computes the min value for a set of coefficients (gammas)


    Parameters
    ----------
    coeffs: :py:class `~xarray.DataArray`
        coefficients for the gammas used to compute the analytic min

    min_max: list
        min and max temp values to evaluate derivative at

    gammas_min_path: str
        path to save/read data

    Returns
    -------
        Dataset
            :py:class `~xarray.DatArray` min temp by hierid

    Example
    -------


    '''
    minx = min_max.min()
    maxx = min_max.max()

    derivcoeffs = np.array(coeffs[1:]) * np.arange(1, len(coeffs)) # Construct the derivative
    roots = np.roots(derivcoeffs[::-1])

    # Filter out complex roots; note: have to apply real_if_close to individual values, not array until filtered
    possibles = filter(lambda root: np.real_if_close(root).imag == 0 and np.real_if_close(root) >= minx and np.real_if_close(root) <= maxx, roots)
    possibles = list(np.real_if_close(possibles)) + [minx, maxx]

    with warnings.catch_warnings(): # catch warning from using infs
        warnings.simplefilter("ignore")
        values = np.polyval(coeffs[::-1], np.real_if_close(possibles))
        
    # polyval doesn't handle infs well
    if minx == -np.inf:
        if len(coeffs) % 2 == 1: # largest power is even
            values[-2] = -np.inf if coeffs[-1] < 0 else np.inf
        else: # largest power is odd
            values[-2] = np.inf if coeffs[-1] < 0 else -np.inf
            
    if maxx == np.inf:
        values[-1] = np.inf if coeffs[-1] > 0 else -np.inf
    
    index = np.argmin(values)

    return possibles[index]

def compute_M_tstar(gammas_ds, min_max, min_function=None, write_path=None):
    '''
    Computes the impact tstar based on that computes m_tstar

    Parameters
    ----------
    gammas_ds: :py:class `~xarray.Dataset` 
        coefficients by hierid Dataset or np.ndarray

    min_max: np.array
        values to evaluate min at

    min_function: minimizing function to compute tstar


    write_path: str

    Returns
    -------

    M_star Dataset
        :py:class`~xarray.Dataset` of impacts evaluated at tstar. 


    .. note:: writes to disk and subsequent calls will read from disk. 
    '''

    if os.path.isdir(write_path):
        return get_mstar(write_path)


    tstar = np.apply_along_axis(
                        lambda x: min_function(x min_max),
                        1, 
                        gammas_ds.values)

    #Something like evaluate the structure of gammas_ds and generate a temperature dataset.

    tas_star_something= xr.Dataset()
    for i, pred in enumerate(gammas_ds):
        tas_star_something['tstar_{}'.format(i)] = xr.DataArray(tstar**i, coords={gammas_ds['hierid']}, dims=['hierid'])


    #Compute M_star as a function of tas_star_something and gammas_ds
    M_star = tas_star_something*gammas_ds

    if not os.path.isdir(os.path.dirname(write_path)):
            os.path.makedir(os.path.dirname(write_path))

    M_star.to_netcdf(write_path)

    return M_star

@memoize
def get_mstar(path):
    return xr.open_dataset(path)

