import xarray as xr
import pandas as pd


class Impact(gammas, weather, covariates):
  '''
    Base class for computing an impact as specified by the Climate Impact Lab

  '''
  def __init__(self):
    self.gammas = gammas
    self.spec = spec
    self.annual_weather = self.get_annual_weather(weather, self.spec)
    self.betas = self.comput_betas(self.gammas, covariate)


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

  def compute_betas(gammas, spec, covariates):
    '''
    Computes the matrices beta*gamma x IR for each covariates 
    
    Parameters
    ----------
    gammas: :py:class `~xarray.Dataset`
        Coefficients for pred/covar combo

    spec: 
      specifies the shape/structure of computation

    clim_covars: :py:class `~xarray.Dataset`
        covariates for each hierid
    
    gpd_covars: :py:class `~xarray.Dataset`
        covariates for each hierid
 
    Returns
    -------
      :py:class `~xarray.Dataset` values for each predname beta

    '''

    #Something like this where we use spec or sympy to structure a computation
    # betas = xr.Dataset()
    # for pred, pred_covar in spec.items():
    #       betas[pred] = gammas[pred] + (gammas[pred_covar]*covar for covar in covars)


    betas = xr.Dataset()

    betas['tas'] = (gammas['beta0_pow1'] + gammas['gdp_pow1'] * gdp_covar['gdppc'] + gammas['tavg_pow1']*clim_covar['tas'])
    betas['tas-poly-2'] = (gammas['beta0_pow2'] + gammas['gdp_pow2'] * gdp_covar['gdppc'] + gammas['tavg_pow2']*clim_covar['tas'])
    betas['tas-poly-3'] = (gammas['beta0_pow3'] + gammas['gdp_pow3'] * gdp_covar['gdppc'] + gammas['tavg_pow3']*clim_covar['tas'])
    betas['tas-poly-4'] = (gammas['beta0_pow4'] + gammas['gdp_pow4'] * gdp_covar['gdppc'] + gammas['tavg_pow4']*clim_covar['tas'])

    return betas


  def compute(gammas, 
                      spec,  
                      gdp_covars,
                      clim_covars,
                      annual_weather_paths,
                      min_max,boundaries
                      baseline,
                      impact_function=None):
    '''
    Computes an impact for a unique set of gdp, climate, weather and gamma coefficient inputs.
    For each set of these, we take the analytic minimum value between two points and 
    

    '''
    #Generate Betas
    betas = self.compute_betas(gammas, spec, clim_covars, gdp_covars)

    #Compute the min for flat curve adaptation
    clipped_curve_mins = self.find_mins(betas, min_max_boundaries)

    #Compute Raw Impact
    impact_raw = impact_function(betas, self.annual_weather, spec)

    #Compare values and evaluate a max
    impact_clipped = np.maximum(impact_raw, clipped_curve_mins)

    #Sum to annual, substract baseline, normalize 
    impact_annual = (impact_clipped.sum(dim='time')  - baseline['baseline'])/1e5

    return impact_annual


def find_mins(betas, min_max_boundaries):
  raise NotImplementedError


  def postprocess_daily(impact):
    raise NotImplementedError

  def postprocess_annual(impact):
    raise NotImplementedError
