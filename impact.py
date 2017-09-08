'''
Base class for computing an impact as specified by the Climate Impact Lab
'''


class Impact(gammas_file, weather, covariates):
  '''
    


  '''



  def _read_csvv(gammas_file):



  def prep_gammas(gammas):



    



    return gammas


  def prepare_annual_weather(weather):

    return weather

  def compute_betas(gammas, covariates):

    return betas


  def polynomial(betas, weather):


    return impact_polynomial


  def spline(betas, weather):

    return impact_spline


  def compute_impact(betas, weather, impact=None):


    return impact

  def postprocess_daily(impact):
    return

  def postprocess_annual(impact):
    return
