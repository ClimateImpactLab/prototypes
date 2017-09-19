import os
import xarray as xr
import numpy as np
import warnings
from impact import Impact




def _findpolymin(coeffs, min_max):
    '''
    Computes the min value `t_star` for a set of coefficients (gammas)
    for a polynomial damage function


    Parameters
    ----------
    coeffs: :py:class `~xarray.DataArray`
        coefficients for the gammas used to compute the analytic min

    min_max: list
       min and max temp values to evaluate derivative at

    Returns
    -------
        int: t_star

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

def minimize_polynomial(da, dim='prednames', bounds=None):
    '''
    Constructs the t_star-based weather data array by applying `np.apply_along_axis`
    to each predictor dimension and construcing data variables up to the order specified in `prednames`

    Parameters
    ----------
    da: DataArray
        :py:class:`~xarray.DataArray` of betas by hierid by predname by outcome

    dim: str
        dimension to evaluate the coefficients at

    bounds: list
        values to evaluate between

    Returns
    -------
    DataArray
        :py:class:`~xarray.DataArray` of reconstructed weather at t_star

    '''
    t_star_values = np.apply_along_axis(_findpolymin, da.get_axis_num(dim), da.values, min_max = bounds)

    if t_star_values.shape != tuple([s for i, s in enumerate(da.shape) if i != da.get_axis_num(dim)]):
        raise ValueError('_findpolymin returned an unexpected shape: {}'.format(t_star_values.shape))

    t_star = xr.DataArray(
        t_star_values,
        dims=tuple([d for d in da.dims if d != dim]),
        coords={c: da.coords[c] for c in da.coords.keys() if c != dim})

    t_star = t_star.expand_dims(dim, axis=da.get_axis_num(dim))

    # this is the only part I'm unsure of. Should the 0th term be included?
    t_star_poly = xr.concat([t_star**i for i in range(len(da.coords[dim]))], dim=da.coords[dim])

    return t_star_poly
