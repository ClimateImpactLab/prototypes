import os
import numpy as np
from impact import Impact
from baselines import minimize_polynomial
from toolz import memoize
import xarray as xr



class Mortality_Polynomial(Impact):
	'''
	Mortality specific 
	'''
	
	min_function = minimize_polynomial



	def impact_function(self, betas, weather):
		'''
		computes the dot product of betas and annual weather by outcome group

		'''
		#slick
		#sum((betas*annual_weather).data_vars.values()).sum(dim='prednames')
		
		#verbose
		impact =  (betas.sel(prednames='tas')*weather['tas'] + 
                betas.sel(prednames='tas-poly-2')*weather['tas-poly-2'] + 
                betas.sel(prednames='tas-poly-3')*weather['tas-poly-3'] + 
                betas.sel(prednames='tas-poly-4')*weather['tas-poly-4'])

		return impact

	def compute_m_star(self, betas, min_max_boundary=None, t_star_write_path=None):
	    if not os.path.isfile(t_star_write_path):

	      #Compute t_star according to min function
	      	t_star = minimize_polynomial(betas, min_max_boundary)
	      #write to disk
	    	if not os.path.isdir(os.path.dirname(t_star_write_path)):
	              os.path.makedir(os.path.dirname(t_star_write_path))

	    	t_star.to_netcdf(t_star_write_path)

	  	#Read from disk
	  	t_star = self._get_t_star(t_star_write_path)

	  	return sum((t_star*betas).data_vars.values()).sum(dim='prednames')

	@memoize
	def _get_t_star(self, path):
		with xr.open_dataset(path) as ds:
			ds.load()
		return ds
  		


