
import csv
from toolz import memoize
import xarray as xr
import pandas as pd


class Gammas():
	'''
	Base class for reading csvv files. 
	1. Constructs data structure representing covar/predname coefficients
	2. Draws samples for monte carlo
	3. Enables specification for impact function
	'''


	def __init__(self, csvv_path):
		self.csvv_path = csvv_path
		self.data = self._read_csvv(self.csvv_path)



	@memoize
	def _read_csvv(self, csvv_path):
	    '''
	    Returns the gammas and covariance matrix 
	    
	    Parameters
	    ----------
	    path: str
	        path to csvv file

	    Returns
	    -------
	    dict with keys of gamma, gammavcv, prednames, covarnames and residvcv

	  	Extracts necessary information to specify an impact function
	    '''

	    data = {}

	    with open(csvv_path, 'r') as file:
	        reader = csv.reader(file)
	        for row in reader:
	            if row[0] == 'gamma':
	                data['gamma'] = [float(i) for i in reader.next()]
	            if row[0] == 'gammavcv':
	                data['gammavcv'] = [float(i) for i in reader.next()]
	            if row[0] == 'residvcv':
	                data['residvcv'] = [float(i) for i in reader.next()]
	            if row[0] == 'prednames':
	                data['prednames'] = list(set([pred.strip() for pred in reader.next()]))
	            if row[0] == 'covarnames':
	                data['covarnames'] = list(set([cv.strip() for cv in reader.next()]))


	    return data


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


		return self._prep_gammas(self.data)


	def sample(self, seed=None):
		'''
		Takes a draw from a multivariate distribution and returns a Dataset of coefficients. 
		Labels on coefficients can be used to construct a specification of the functional form.


		Parameters
		----------
		seed: int
			number to use to intialize randomization


		Returns: 

			:py:class `~xarray.Dataset` of gamma coefficients organized by covar and pred
		'''


		return self._prep_gammas(seed=seed)




	def _prep_gammas(self, data, seed=None):
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

	    ##########################
	    # Read in data from csvv #
	    ##########################
	    len_preds = len(data['prednames'])  

	    if seed:
	        np.random.seed(seed)
	        data = mn.rvs(data['gamma'], data['gammavcv'])

	    else: 
	        data = data['gamma']

	    gammas = xr.Dataset()
	    indices = {'outcomes': pd.Index(['age0-4', 'age5-64', 'age65+'], name='age')}
	   # print(len(data['prednames']))

	    ##This is wrong, the indices are misaligned and assigning values incorrectly
	    for pwr in range(1, len_preds+1):
	            gammas['beta0_pow{}'.format(pwr)] = xr.DataArray(
	                data[0*pwr::12], dims=('age',), coords={'age':indices['age_cohorts']})
	            gammas['gdp_pow{}'.format(pwr)] = xr.DataArray(
	                data[1*pwr::12], dims=('age',), coords={'age': indices['age_cohorts']})
	            gammas['tavg_pow{}'.format(pwr)] = xr.DataArray(
	                data[2*pwr::12], dims=('age',), coords={'age': indices['age_cohorts']})

	    return gammas