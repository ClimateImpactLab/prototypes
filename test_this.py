import xarray as xr, pandas as pd, numpy as np
from mortality import Mortality_Polynomial
from csvv import Gammas
from impact import Impact
from base import BaseImpact
import time

#location of gammas file
gammas_path = '/global/scratch/jsimcock/data_files/covars/global_interaction_Tmean-POLY-4-AgeSpec.csvv'

#location of weathe for a given year
weather_path = '/global/scratch/mdelgado/projection/gcp/climate/hierid/popwt/daily/{pred}/{scenario}/{model}/{year}/1.5.nc4'

#location for 
t_star_path = '/global/scratch/jsimcock/data_files/covars/t_star_median.nc'

#gdp_covar_path
gdp_covar_path = '/global/scratch/jsimcock/data_files/covars/ssp_kernel_13_gdppc/SSP1/high/{year}/0.1.0.nc'

#clim_covar_path
clim_covar_path = '/global/scratch/jsimcock/data_files/covars/climate/hierid/popwt/tas_kernel_30/{scenario}/{model}/{year}/0.1.1.nc4'

#baseline path
base_path = '/global/scratch/jsimcock/gcp/impacts/mortality-daily/median/historical/low/SSP1/ACCESS1-0/baseline/0.1.1.nc'

#years to compute basline from
base_years = [2000, 2010]

#dict to populate path values
metadata = {'scenario': 'rcp85', 
				'model': 'ACCESS1-0',
				'year': 2015
				}


def demo_run():
	'''
	Demonstration of an atomic impact compute operation. All operations are parametrized with 
	variables like rcp scenario, ssp, econ_model, year, etc. This allows us to simplify the entire process

	This will not simulate any adaptation scenarios. This is suggestive of a full-adaptation scenario.
	The baseline year 2015 is shown to demonstrate how baseline computation interacts with computation of any given year. 

	

	This demonstration performs the following routine

	1. Initialize a gammas object from a csvv file
	2. Loads single-year covariate data variables according to parameters in metadata 
	3. Takes the sample from the gammas distribution for each covariate/predname 
	4. Initializes the BaseImpact object to compute/read baseline impact data. Baseline
		data is generated only once per model, mc run, rcp, ssp, econ_model combination. 
		Subsequent operation are read only. 
	5. We then initialize our Mortality_Polynomial class with the weather, prednames and metadata values. Weather
		values for that year are constructed on initialization as they are consistent across the adaptation scenarios.
	6. The main method on each Impact class is `compute`. You call this to run the entire sequence of operations. 
	7. Below each of the sub-routines is specified for clarity. (hopefully its clear)
		a. get_weather: builds a dataset of annual weather based on spec in the csvv/gammas file
		b. Constructs a betas dataarray based on covariates and gammas
		c. Computes the impact functions, based on spec, used for subsequent comparison with m_star
		d. Use betas to compute t_star and m_star
		e. Compute impact_minned as min between impact and m-star
		f. Sum to annual impact
		g. subtract baseline from annual impact to get rebased value
	
	'''

	t1  = time.time()

	#We need to initialize the gammas object
	g = Gammas(gammas_path)

	#initialize covars
	gdp = xr.open_dataarray(gdp_covar_path.format(year=metadata['year']))
	climtas = xr.open_dataarray(clim_covar_path.format(**metadata))
	
	#take a median draw from dist
	gammas = g.median()
	
	#initialize baseline impact with weather, gammas, and base_years
	base = BaseImpact(weather_path, gammas.prednames.values, metadata, base_years, base_path)

	#compute baseline for median run
	base_median = base.compute(gammas, gdp, climtas)

	#Initialize the Current impact function. Weather is constructed
	m = Mortality_Polynomial(weather_path, gammas.prednames.values, metadata)

	#Compute betas
	betas = m._compute_betas(gammas, gdp, climtas)

	#we have our weather already so we can compute our impact as specified in the module
	#in this case impact_function is (betas*m.weather).sum(dim='prednames')
	impact = m.impact_function(betas, m.weather)

	#we then need to compute t_star and m_star
	m_star = m._compute_m_star(betas, bounds = [10,25], t_star_path=t_star_path)

	#we then compute the min of m_star and impact
	impact_minned = xr.ufuncs.minimum(impact, m_star)

	#sum to annual
	summed = impact_minned.sum(dim='time')

	#rebase
	rebased = summed - base_median
	t2 = time.time()
	print('snippets: {}'.format(t2-t1))



	#now all at once
	impact = m.compute(gammas, gdp, climtas, base_median, bounds=[10,25], t_star_path=t_star_path) 
	t3 = time.time()
	print('all at once: {}'.format(t3 - t1))


	return rebased, impact



if __name__=='__main__':
	demo_run()                                       


