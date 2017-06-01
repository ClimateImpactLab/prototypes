'''
20-year average polynomial tas specification
'''

import sys, os, re

waterfall_dir = '../../public/waterfall/'
sys.path.append(waterfall_dir)

from waterfall import LocalFall
import itertools

from climate_toolbox import(
    load_bcsd)


BCSD_orig_files = (
    '/global/scratch/jiacany/nasa_bcsd/raw_data/{rcp}/{model}/{variable}/' +
    '{variable}_day_BCSD_{rcp}_r1i1p1_{model}_{year}.nc')

WRITE_PATH = (
    '/global/scratch/mdelgado/web/gcp/climate/{rcp}/{agglev}/{variable}/' +
    '{variable}_{agglev}_{aggwt}_{model}_{pername}.nc')

ADDITIONAL_METADATA = dict(
    description=__doc__.strip(),
    repo='https://github.com/ClimateImpactLab/pipelines',
    file='/climate/jobs/impactlab-website/bcsd-orig.py',
    execute='climate.jobs.impactlab_website.bcsd_orig.main',
    project='gcp', 
    team='climate',
    weighting='areawt',
    frequency='20yr')

PERIODS = [
    dict(rcp='historical', pername='1986', years=list(range(1986, 2006))),
    dict(rcp='rcp85', pername='2020', years=list(range(2020, 2040))),
    dict(rcp='rcp85', pername='2040', years=list(range(2040, 2060))),
    dict(rcp='rcp85', pername='2080', years=list(range(2080, 2100)))]

MODELS = list(map(lambda x: dict(model=x), [
    'ACCESS1-0',
    'bcc-csm1-1',
    'BNU-ESM',
    'CanESM2',
    'CCSM4',
    'CESM1-BGC',
    'CNRM-CM5',
    'CSIRO-Mk3-6-0',
    'GFDL-CM3',
    'GFDL-ESM2G',
    'GFDL-ESM2M',
    'IPSL-CM5A-LR',
    'IPSL-CM5A-MR',
    'MIROC-ESM-CHEM',
    'MIROC-ESM',
    'MIROC5',
    'MPI-ESM-LR',
    'MPI-ESM-MR',
    'MRI-CGCM3',
    'inmcm4',
    'NorESM1-M']))

AGGREGATIONS = [
    {'agglev': 'ISO', 'aggwt': 'areawt'},
    {'agglev': 'hierid', 'aggwt': 'areawt'}]


def polynomials(ds, powers):
    '''
    Raises all data variables to all values in ``powers``

    Parameters
    ----------

    ds : xarray.Dataset
        Dataset to compute powers from. All data_vars will be
        raised

    powers: 

    '''

    for power in powers:
        if power < 2:
            continue

        for var in ds.data_vars.keys():
            ds[var + '_{}'.format(power)] = ds[var]**power

    yield ds


def bcsd_finder(**job):
    return BCSD_orig_files.format(**job)


def load_bcsd_test(fp, *args, **kwargs):
    import xarray as xr
    import numpy as np
    import pandas as pd

    varname = re.search(r'\/(?P<var>tas(min|max)?)\/', fp).group('var')
    year = re.search(r'_(?P<year>[0-9]{4})(_[A-Z]{3})?\.nc', fp).group('year')

    yield xr.Dataset({
        varname: xr.DataArray(
            np.arange(1000).reshape(10,10,10),
            dims=('lat', 'lon', 'time'),
            coords={
                'lat': np.linspace(-180, 180, 10),
                'lon': np.linspace(-180, 180, 10),
                'time': pd.date_range('{}-1-1'.format(year), periods=10, freq='D')
            })
        })

def regional_aggregation(ds):
    ds = ds.groupby(ds.lat//20).mean()
    yield ds

def output(ds, pattern, job):
    print(pattern.format(**job))
    ds.to_netcdf(pattern.format(**job))

    yield

def iterate_jobs(*args):
    for job in itertools.product(*args):
        spec = {}
        for component in job:
            spec.update(component)
        yield spec

def main():
    job = (LocalFall(os.path.expanduser('~/pipes'))
        .pipe(iterate_jobs, PERIODS, MODELS, AGGREGATIONS)
        .save('job')
        .nest(
            LocalFall(os.path.expanduser('~/pipes'))
                .save('job')
                .pipe(lambda job: [bcsd_finder(year=year, variable='tas', **job) for year in job['years']])
                .pipe(load_bcsd_test, broadcast_dims=('time',))
                .pipe(polynomials, [1,2,3,4,5])
                .pipe(lambda ds: [ds.mean()]))
        .pipe(regional_aggregation)
        .pipe(output, WRITE_PATH, retrieve='job'))

    job.run()

if __name__ == '__main__':
    main()