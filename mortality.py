from impact import Impact
from mins import minimize_polynomial
import xarray as xr
import numpy as np


class Mortality_Polynomial(Impact):
	'''
	Mortality specific 
	'''

	def impact_function(self, betas, weather):
		'''
		computes the dot product of betas and annual weather by outcome group

		'''
		#slick
		impact = (betas*weather).sum(dim='prednames')
		
		#verbose
		# impact =  (betas.sel(prednames='tas')*weather['tas'] + 
  #               betas.sel(prednames='tas-poly-2')*weather['tas-poly-2'] + 
  #               betas.sel(prednames='tas-poly-3')*weather['tas-poly-3'] + 
  #               betas.sel(prednames='tas-poly-4')*weather['tas-poly-4'])



		return impact

	def compute_m_star(self, betas, min_max_boundary=None, t_star_path=None):
		if not os.path.isfile(t_star_path):

        	#Compute t_star according to min function
    		t_star = minimize_polynomial(betas, min_max_boundary)

    	if not os.path.isdir(os.path.dirname(t_star_path)):
   	 		os.makedirs(os.path.dirname(t_star_path))

    	t_star.to_netcdf(t_star_path)

    	#Read from disk
		t_star = self._get_t_star(t_star_path)
		print(t_star)
		print(betas)
      
		return sum((t_star*betas).data_vars.values()).sum(dim='prednames')
  		


