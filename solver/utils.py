"""
This file contains the functions used in the solver.
"""

import numpy as np
import math

def lambda0(k, w, config):
    xi0 = config['xi0']
    nu0 = config['nu0']
    mask = k**2 >= xi0/nu0*w**2
    return (1*mask + 1j*(1-mask))*np.sqrt((k**2 - xi0/nu0*w**2)*mask + (xi0/nu0*w**2 - k**2)*(1 - mask))

def lambda1(k, w, config):
    
    xi1 = config['xi1']
    nu1 = config['nu1']
    a = config['a']
    
    shared = np.sqrt((k**2 - xi1/nu1*w**2)**2 + (a*w/nu1)**2)
    
    return 1/math.sqrt(2)*np.sqrt(k**2 - xi1/nu1*w**2 + shared) - 1j/math.sqrt(2)*np.sqrt(xi1/nu1*w**2 - k**2 + shared)

def inv_f(x, l0, config):
    
    """
    Computes the inverse of the function f to avoid overflow.
    """
    
    L = config['L']
    nu0 = config['nu0']
    
    return np.exp(-l0*L)*1/((l0*nu0 - x)*np.exp(-2*l0*L) + (l0*nu0 + x))

def ki(k, w, alpha, l1, l0, ft_g, config):
    nu1 = config['nu1']
    nu0 = config['nu0']
    return ft_g(k, w)*(inv_f(l1*nu1, l0, config)*(l0*nu0 - l1*nu1) - inv_f(alpha, l0, config)*(l0*nu0 - alpha))

def gamma(k, w, alpha, l1, l0, ft_g, config):
    nu1 = config['nu1']
    nu0 = config['nu0']
    return ft_g(k, w)*(inv_f(l1*nu1, l0, config)*(l0*nu0 + l1*nu1) - inv_f(alpha, l0, config)*(l0*nu0 + alpha))