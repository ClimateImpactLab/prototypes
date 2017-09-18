import xarray as xr, pandas as pd, numpy as np
from mortality import Mortality_Polynomial
from csvv import Gammas
from impact import Impact
import time



def test_this():

	t1  = time.time()
	g = Gammas('/global/scratch/jsimcock/data_files/covars/global_interaction_Tmean-POLY-4-AgeSpec.csvv')
	gdp = xr.open_dataset('/global/scratch/jsimcock/data_files/covars/ssp_kernel_13_gdppc/SSP1/high/2015/0.1.0.nc')
	climtas = xr.open_dataset('/global/scratch/jsimcock/data_files/covars/climate/hierid/popwt/tas_kernel_30/rcp45/ACCESS1-0/2015/0.1.1.nc4')
	gdp = gdp.rename({'gdppc': 'loggdppc'})
	climtas = climtas.rename({'tas':'climtas'})


	path = '/global/scratch/mdelgado/projection/gcp/climate/hierid/popwt/daily/{pred}/{scenario}/{model}/{year}/1.5.nc4'
	t_star_path = '/global/scratch/jsimcock/data_files/covars/t_star_median.nc'

	gammas = g.median()
	metadata = {'scenario': 'rcp85', 
				'model': 'ACCESS1-0',
				'year': 2015
				}

	m = Mortality_Polynomial(path, gammas.prednames.values, metadata)
	# betas = m._compute_betas(gammas, [gdp, climtas])


	impact = m.compute(gammas, gdp, climtas, bounds=[10,25], t_star_path=t_star_path) 
	baseline = xr.open_dataset('/global/scratch/jsimcock/gcp/impacts/mortality-daily/median/historical/low/SSP1/ACCESS1-0/baseline/0.1.1.nc')


	t2 = time.time()

	print(t2-t1)

	return impact



if __name__=='__main__':
	test_this()                                       


