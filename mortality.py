from impact import Impact
from mins import minimize_polynomial
import xarray as xr
import numpy as np
import os

class Mortality_Polynomial(Impact):
	'''
	Mortality specific Impact spec
	'''
	@staticmethod
	def min_function(*args, **kwargs):
		'''
		helper function to call minimization function for given mortality polynomial spec
		mortality_polynomial implements findpolymin through `np.apply_along_axis`

		Parameters
		----------

		betas: DataArray
			:py:class:`~xarray.DataArray` of hierid by predname by outcome

		dim: str
			dimension to apply minimization to

		bounds: list
			values between which to search for t_star

		Returns
		-------
			:py:class:`~xarray.DataArray` of hierid by predname by outcome

		.. note:: overides `min_function` in Impact base class
		'''

		return minimize_polynomial(*args, **kwargs)

	def impact_function(self, betas, weather):
		'''
		computes the dot product of betas and annual weather by outcome group

		Parameters
		----------

		betas: DataArray
			:py:class:`~xarray.DataArray` of hierid by predname by outcome

		weather: DataArray
			:py:class:`~xarray.DataArray` of hierid by predname by outcome

		Returns
		-------
		DataArray
			:py:class:`~xarray.DataArray` of impact by outcome by hierid

		.. note:: overrides `impact_function` method in Impact base class
		'''
		
		#slick
		impact = (betas*weather).sum(dim='prednames')

		#verbose
		# impact =  (betas.sel(prednames='tas')*weather['tas'] + 
  #               betas.sel(prednames='tas-poly-2')*weather['tas-poly-2'] + 
  #               betas.sel(prednames='tas-poly-3')*weather['tas-poly-3'] + 
  #               betas.sel(prednames='tas-poly-4')*weather['tas-poly-4'])

		return impact


 	####################
 	# to be deprecated #
 	####################  
 	def _construct_covars(self, gdp_covar, clim_covar):
    	'''
    	Helper function to construct the covariates dataarray

    	Parameters
    	-----------
    	gdp_covar: :py:class:`~xarray.DataArray`
    	  hierid by gdp dataarray

    	clim_covar: :py:class:`~xarray.DataArray`
    	  hierid by clim dataarray
	
    	Returns
    	-------
    	  Dataarray
	     	   heirid by covars (1, gdp, clim) :py:class:`~xarray.DataArray`
	    '''

    	#this will change once we generalize
    	gdp_covar = gdp_covar.drop('iso')
    	gdp = gdp_covar.rename('loggdppc')
    	climtas = clim_covar.rename('climtas')


    	ones = xr.DataArray(np.ones(len(gdp.hierid)), coords={'hierid': gdp.hierid}, dims=['hierid'], name='1')
    	cv = [ones, gdp, climtas]
    	covars = xr.concat(cv, pd.Index([i.name for i in cv], name='covarnames'))
    	return covars

	####################
 	# to be deprecated #
 	####################  
	def compute_betas(self, gammas, gdp_covar, clim_covar):
    	'''
    	Computes the matrices beta*gamma x IR for each covariates 
    
    	Parameters
    	----------
    	gammas: :py:class `~xarray.Dataset`
        	Coefficients for pred/covar combo

    	gdp_covar: :py:class:`~xarray.DataArray`
      		hierid by gdp dataarray

    	clim_covar: :py:class:`~xarray.DataArray`
      		hierid by clim dataarray    
 
    	Returns
    	-------
      	Dataarray
        	hierid by outcome and predname :py:class `~xarray.Dataarray` 
    	'''
    	covars = self._construct_covars(gdp_covar, clim_covar)
    
    	betas = (gammas*covars).sum(dim='covarnames')

    	betas = beta_vars['1'] + beta_vars['climtas'] + beta_vars['loggdppc']

    	return betas






