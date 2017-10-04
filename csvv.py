
import csv
import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mn


def get_gammas(csvv_path):
    '''
    Returns the gammas and covariance matrix 
    
    Parameters
    ----------
    path: str
        path to csvv file

    Returns
    -------
    dict with keys of gamma, gammavcv, prednames, covarnames outcomes, and residvcv

  	Extracts necessary information to specify an impact function
    '''

    data = {}

    with open(csvv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'gamma':
                data['gamma'] = np.array([float(i) for i in reader.next()])
            if row[0] == 'gammavcv':
                data['gammavcv'] = np.array([float(i) for i in reader.next()])
            if row[0] == 'residvcv':
                data['residvcv'] = np.array([float(i) for i in reader.next()])
            if row[0] == 'prednames':
                data['prednames'] = [i.strip() for i in reader.next()]
            if row[0] == 'covarnames':
                data['covarnames'] = [i.strip() for i in reader.next()]
            if row[0] == 'outcome': 
            	data['outcome'] =[cv.strip() for cv in reader.next()]

    index = pd.MultiIndex.from_tuples(zip(data['outcome'], data['prednames'], data['covarnames']), 
	    									names=['outcome', 'prednames', 'covarnames'])

    g = Gammas(data['gamma'], data['gammavcv'], index)

    return g 



class Gammas(object):
	'''
	Base class for reading csvv files. 
	1. Constructs data structure representing covar/predname coefficients
	2. Draws samples for monte carlo
	'''

	def __init__(self, gammas, gammavcv, index):
		'''	
		Constructor for gammas object

		Parameters
		----------
		gammas: array 
			:py:class:`~numpy.array`
			point estimates of median values for multivariate distribution
		
		gammavcv: array
			:py:class:`~numpy.array`
			covariance matrix for point estimates of median values for multivariate distribution

		index: MultiIndex
			:py:class:`~pandas.MultiIndex` of prednames, covarnames and outcomes


		Returns
		-------
		DataArray	
			:py:class `~xarray.DataArray`  

		'''
		self.gammas = gammas
		self.gammavcv = gammavcv
		self.index = index

	def median(self):
		'''
		Returns the values in the array of gammas organized according to specification
		
		Parameters
		----------
		None

		Returns
		-------

			:py:class `~xarray.DataArray` of gamma coefficients organized by covar and pred
		'''

		return self._prep_gammas()

	def sample(self, seed=None):
		'''
		Takes a draw from a multivariate distribution and returns a Dataset of coefficients. 
		Labels on coefficients can be used to construct a specification of the functional form.


		Parameters
		----------
		seed: int
			number to intialize randomization


		Returns: array
			:py:class:`~numpy.array` of gamma coefficients

		'''

		return self._prep_gammas(seed=seed)

	def _prep_gammas(self, seed=None):
	    '''
	    Produces the gammas data structure
	
	    Parameters
	    ----------	  
	    seed: int
	        seed for random draw

	    Returns
	    -------
			:py:class `~xarray.DataArray` of gamma coefficients organized by covar and pred
	    
	    '''

	    if seed:
	        np.random.seed(seed)
	        g = mn.rvs(self.gammas, self.gammavcv)

	    else: 
	        g = self.gammas


	    return pd.Series(g, index=self.index).to_xarray()
