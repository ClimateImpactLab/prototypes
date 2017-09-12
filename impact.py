import xarray as xr
import pandas as pd


class Impact(gammas, weather, covariates):
  '''
    Base class for computing an impact as specified by the Climate Impact Lab

  '''
  def __init__(self):
    self.gammas = gammas
    self.spec = spec
    self.annual_weather = self.get_annual_weather(weather, self.gammas.prednames.values)
    self.betas = self.comput_betas(self.gammas, covariate)


  def get_annual_weather(weather, preds):
    '''
    Constructs the annual weather dataset for a given years impact

    Parameters
    ----------
    models_paths: list
        unformatted string representing path to weather variables for that year

    preds: dict
      describes how to format input file paths for weather files

    Returns
    -------

      Dataset
        :py:class `~xarray Dataset` with each weather variable as an `~xarray DataArray`
    '''
    weather_files = [weather.format(pred=pred) for pred in preds]

    datasets = []
    for file in weather_files:
      with xr.open_dataset(file) as ds:
          ds.load()
          datasets.append(ds)

    weather = xr.concat(datasets, pd.Index(preds, name='prednames'))

    return weather

  def compute_betas(gammas, covariates):
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

    #One way to do this is to save them as a list of values, this would preserve their value and then when doing findmins we could 
    #simply concat together in a long array to compute the min
    #after compute min, we could sum then compute the impact

    betas = gammas.sel(covarnames='1') + gammas.sel(covarnames='climtas')*climtas + gammas.sel(covarnames='loggdppc')*gdppc 

    betas = sum((gammas*covars).data_vars.values())

    # betas['tas'] = (gammas['beta0_pow1'] + gammas['gdp_pow1'] * gdp_covar['gdppc'] + gammas['tavg_pow1']*clim_covar['tas'])
    # betas['tas-poly-2'] = (gammas['beta0_pow2'] + gammas['gdp_pow2'] * gdp_covar['gdppc'] + gammas['tavg_pow2']*clim_covar['tas'])
    # betas['tas-poly-3'] = (gammas['beta0_pow3'] + gammas['gdp_pow3'] * gdp_covar['gdppc'] + gammas['tavg_pow3']*clim_covar['tas'])
    # betas['tas-poly-4'] = (gammas['beta0_pow4'] + gammas['gdp_pow4'] * gdp_covar['gdppc'] + gammas['tavg_pow4']*clim_covar['tas'])

    return betas


  def compute(gammas, 
                      spec,  
                      gdp_covars,
                      clim_covars,
                      annual_weather_paths,
                      min_max,boundaries
                      baseline,
                      min_function=None,
                      min_write_path=None,
                      impact_function=None):
    '''
    Computes an impact for a unique set of gdp, climate, weather and gamma coefficient inputs.
    For each set of these, we take the analytic minimum value between two points and 
    

    '''
    #Generate Betas
    betas = self.compute_betas(gammas, spec, clim_covars, gdp_covars)

    #Compute Raw Impact
    impact= impact_function(betas, self.annual_weather, spec)

    #Compute the min for flat curve adaptation
    if min_function:
      clipped_curve_mins = self.min_function(gammas, min_max_boundaries, min_write_path)
      #Compare values and evaluate a max
      impact = np.maximum(impact, clipped_curve_mins)

    


    #Sum to annual, substract baseline, normalize 
    impact_annual = (impact.sum(dim='time')  - baseline['baseline'])

    return impact_annual


def find_mins(betas, min_max_boundaries):
  raise NotImplementedError


  def postprocess_daily(impact):
    raise NotImplementedError

  def postprocess_annual(impact):
    raise NotImplementedError
