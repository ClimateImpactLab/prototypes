import os
import numpy as np
from impact import Impact
from baselines import _findpolymin
from toolz import memoize



class Mortality_Polynomial(Impact):
	'''
	Mortality specific 
	'''


	def _impact_function(self, betas, weather):
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

	def _compute_m_star(betas, min_function=_findpolymin, min_max=[10,25], write_path=None):
	    '''
	    Computes m_star, the value of an impact function for a given set of betas given t_star. 
	    t_star, the value t at which an impact is minimized for a given hierid is precomputed 
	    and used to compute m_star.

	    Parameters
	    ----------
	    betas: :py:class `~xarray.Dataset` 
	        coefficients by hierid Dataset 

	    min_max: np.array
	        values to evaluate min at 

	    min_function: minimizing function to compute tstar

	    write_path: str

	    Returns
	    -------

	    m_star Dataset
	        :py:class`~xarray.Dataset` of impacts evaluated at tstar. 


	    .. note:: writes to disk and subsequent calls will read from disk. 
	    '''
	    #if file does not exist create it
	    if not os.path.isfile(write_path):

	        #Compute t_star according to min function
	        t_star = np.apply_along_axis(min_function, 1, betas, min_max)

	        #Compute the weather dataset with t_star as base weather var
	        t_star_poly = xr.Dataset()
	        for i, pred in enumerate(betas.prednames.values):
	            t_star_poly['{}'.format(pred)] = xr.DataArray(t_star**(i+1), 
	                                            coords={'hierid': betas['hierid'], 'outcome': betas['outcome']}, 
	                                            dims=['outcome', 'hierid']
	                                            )

	        #write to disk
	        if not os.path.isdir(os.path.dirname(write_path)):
	                os.path.makedir(os.path.dirname(write_path))

	        t_star_poly.to_netcdf(write_path)

	    #Read from disk
	    t_star_poly = _get_t_star(write_path)

	    
	    return sum((t_star_poly*betas).data_vars.values()).sum(dim='prednames')

	@memoize
	def _get_t_star(path):
	    '''
	    Returns cached verison of t_star

	    '''
	    with xr.open_dataset(path) as ds:
	        ds.load()
	    return ds



