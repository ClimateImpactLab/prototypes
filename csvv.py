
import csv
import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mn


def read_csvv(csvv_path):
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

    return data



class Gammas(object):
	'''
	Base class for reading csvv files. 
	1. Constructs data structure representing covar/predname coefficients
	2. Draws samples for monte carlo
	3. Enables specification for impact function
	'''

	def __init__(self, data):
		'''	
		Constructor for gammas object

		Parameters
		----------
		data: dict
			dictionary specifying covarnames, prednames, gammas, cov matrix etc.

		Returns
		-------
		DataArray	
			:py:class `~xarray.DataArray`  

		'''
		self.data = data

	def median(self):
		'''
		Returns the values in the array of gammas organized according to specification
		
		Parameters
		----------
		None

		Returns
		-------

			:py:class `~xarray.Dataset` of gamma coefficients organized by covar and pred
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


		Returns: 

			:py:class `~xarray.Dataset` of gamma coefficients organized by covar and pred
		'''

		return self._prep_gammas(seed=seed)

	def _prep_gammas(self, seed=None):
	    '''
		Constructs the data structure to organize the functional spec of impact computation. 
		If seed is provided a random sample is drawn from the multivariate distribution. 


	    Parameters
	    ----------
	    path: str
	        path to file 

	    power: int
	      for polynomial specifications

	    seed: int
	        seed for random draw

	    Returns
	    -------
	    Xarray Dataset
	    '''

	    if seed:
	        np.random.seed(seed)
	        g = mn.rvs(self.data['gamma'], self.data['gammavcv'])

	    else: 
	        g = self.data['gamma']


	    ind = pd.MultiIndex.from_tuples(zip(self.data['outcome'], self.data['prednames'], self.data['covarnames']), 
	    									names=['outcome', 'prednames', 'covarnames'])

	    gammas = pd.Series(g, index=ind)

	    gammas = xr.DataArray.from_series(gammas)

	    return gammas
