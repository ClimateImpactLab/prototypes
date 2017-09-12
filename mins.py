import warnings
import numpy as np

def findpolymin(coeffs, minx, maxx):
    """coeffs should start from the intercept and go to the highest order.
    That is, the polynomial is sum_k coeffs[k] * x^k"""

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

if __name__ == '__main__':
    print findpolymin([0, 0, -3, 1, 2], -np.inf, np.inf)
    print findpolymin([0, 0, -3, 1, 2], 0, np.inf)
    print findpolymin([0, 0, -3, 1, 2], -1, 1)