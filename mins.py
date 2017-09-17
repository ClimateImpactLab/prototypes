import xarray as xr
import numpy as np
from toolz import memoize
import warnings
import os
from impact import Impact




def _findpolymin(coeffs, min_max):
    '''
    Computes the min value for a set of coefficients (gammas)


    Parameters
    ----------
    coeffs: :py:class `~xarray.DataArray`
        coefficients for the gammas used to compute the analytic min

    min_max: list
       min and max temp values to evaluate derivative at

    gammas_min_path: str
        path to save/read data

    Returns
    -------
        Dataset
            :py:class `~xarray.DatArray` min temp by hierid

    Example
    -------
    '''
   
    minx = np.asarray(min_max).min()
    maxx = np.asarray(min_max).max()

    derivcoeffs = np.array(coeffs[1:]) * np.arange(1, len(coeffs)) # Construct the derivative
    roots = np.roots(derivcoeffs[::-1])

    # Filter out complex roots; note: have to apply real_if_close to individual values, not array until filtered
    possibles = filter(lambda root: np.real_if_close(root).imag == 0 and np.real_if_close(root) >= minx and np.real_if_close(root) <= maxx, roots)
    possibles = list(np.real_if_close(possibles)) + [minx, maxx]

    with warnings.catch_warnings(): # catch warning from using infs
        warnings.simplefilter("ignore")
    
        values = np.polyval(coeffs[::-1], np.real_if_close(possibles))  
        
    # polyval doesn't handle infs well
    if minx == -np.inf:
        if len(coeffs) % 2 == 1: # largest power is even
            values[-2] = -np.inf if coeffs[-1] < 0 else np.inf
        else: # largest power is odd
            values[-2] = np.inf if coeffs[-1] < 0 else -np.inf
            
    if maxx == np.inf:
        values[-1] = np.inf if coeffs[-1] > 0 else -np.inf
    
    index = np.argmin(values)

    return possibles[index]

def minimize_polynomial(da, dim='prednames', bounds=[10,25]):
    '''


    '''
    print(da.dims)
    vals = da.prednames.values
    t_star_values = np.apply_along_axis(_findpolymin, 1, da, min_max=bounds)

    if t_star_values.shape != tuple([s for i, s in enumerate(da.shape) if i != 1]):
        raise ValueError('_findpolymin returned an unexpected shape: {}'.format(t_star_values.shape))

    t_star = xr.DataArray(
        t_star_values,
        dims=tuple(['outcome', 'hierid']),
        coords={'outcome': da.outcome, 'hierid': da.hierid})

    t_star = t_star.expand_dims(dim, axis=1)

    # this is the only part I'm unsure of. Should the 0th term be included?
    t_star_poly = xr.concat([t_star**i for i in range(len(vals))], pd.Index(vals, name=dim))

    return t_star_poly
