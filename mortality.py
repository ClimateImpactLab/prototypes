import os
import numpy as np
from impact import Impact
from baselines import minimize_polynomial
from toolz import memoize



class Mortality_Polynomial(Impact):
	'''
	Mortality specific 
	'''
	self.min_function = minimize_polynomial

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


