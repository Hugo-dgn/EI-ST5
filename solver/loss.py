import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import scipy.optimize as opt

import solver.utils as utils

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) #ignore the warnings because np.log(0) = -inf can happen

def e_k(w, alpha, ft_g, config):
    
    """
    Computes the e_k function. This function returns an array of shape (2*N+1, number of batch, batch size).
    If only one alpha is given, tthe shape of the output is (2*N+1, 1, 1).
    Here, N is int(L/dy)
    """
    
    xi0 = config['xi0']
    nu0 = config['nu0']
    L = config['L']
    A = config['A']
    B = config['B']
    dy = config['dy']
    
    
    N = int(L/dy)
    k = np.linspace(-N*np.pi/L, N*np.pi/L, 2*N+1).reshape(-1, 1, 1)
    k = np.array(k).reshape(-1, 1, 1)
    
    mask = k**2 >= xi0/nu0*w**2 # This mask translates the condition k**2 >= xi0/nu0*w**2 into a boolean array.
    
    #compute the important quantities
    l0 = utils.lambda0(k, w, config)
    l1 = utils.lambda1(k, w, config)
    
    ki = utils.ki(k, w, alpha, l1, l0, ft_g, config)
    gamma = utils.gamma(k, w, alpha, l1, l0, ft_g, config)
    
    n_ki = np.abs(ki)
    n_gamma = np.abs(gamma)
    
    gamma_lambda_exp = np.exp( np.log(n_gamma) + l0*L )**2 #add the exponnent to avoid overflow
    lambda_exp = np.exp(-2*l0*L)
    
    #compute the loss in both cases. The first case is when k**2 >= xi0/nu0*w**2 and the second case is when k**2 < xi0/nu0*w**2.
    #Because all the extensive calculations are done (like exponentials), we can compute both cases and then pick the right one for each k.
    #Only multiplication and addition are done here.
    
    #first case : k**2 >= xi0/nu0*w**2
    part11 = ( A + B*k**2 )*( 1/(2*l0)*( n_ki**2*( 1 - lambda_exp ) + gamma_lambda_exp*( 1 - lambda_exp)) + 2*L*np.real(ki*np.conj(gamma)) )
    part12 = B*l0/2*( n_ki**2*( 1 - lambda_exp ) + gamma_lambda_exp*( 1 - lambda_exp) - 2*B*l0**2*L*np.real(ki*np.conj(gamma)) )
    
    loss1 = part11 + part12
    
    n_l0 = np.abs(l0)
    im_part = np.imag(ki*np.conj(gamma)*(1-np.exp(-2*l0*L)))
    
    #seconde case : k**2 < xi0/nu0*w**2
    part21 = ( A + B*k**2 )*( L*( n_ki**2 + n_gamma**2 ) + 1j/l0*im_part)
    part22 = B*L*n_l0**2*( n_ki**2 + n_gamma**2 ) + 1j*B*l0*im_part
    
    
    loss2 = part21 + part22
    
    #pick the right case for each k
    result = np.where(mask, loss1, loss2)
    
    return result

def loss(w, alpha, ft_g, config):
    """
    Compute the loss function. This function returns an numpy array of shape (number of batch, batch size) which is the same shape as alpha.
    """
    
    l_ek = e_k(w, alpha, ft_g, config)
    
    #sum over k to get the loss
    loss_value = np.abs(np.sum(l_ek, axis=0))
    
    return loss_value

def optim(wmin, wmax, n, ft_g, alpha_max_shearch, eps_shearch, config):
    """
    Compute the optimal alpha for each w in the range [wmin, wmax] with n points.
    """
    optim_alphas = []
    errors = []
    
    W = np.linspace(int(wmin), int(wmax), n)
    
    start = [0, 0]

    for w in tqdm(W):
        
        result = minimize(w, ft_g, alpha_max_shearch, eps_shearch, start, config)
        
        optim_alphas.append(result.x[0] + 1j*result.x[1])
        errors.append(result.fun)
        
        start = result.x

    return optim_alphas, errors

def minimize(w, ft_g, alpha_max_shearch, eps_shearch, start, config):
    """
    Use scipy.optimize.minimize to find the optimal alpha for a given w.
    """
    def f(z):
            z = z[0] + 1j*z[1]
            z = np.array(z).reshape(1, 1, -1)
            l = np.log(loss(w, z, ft_g, config))[0][0]
            return l

    result = opt.minimize(f, start, method='L-BFGS-B', bounds=((-alpha_max_shearch, alpha_max_shearch), (-alpha_max_shearch, alpha_max_shearch)), options={'eps':eps_shearch})
    
    return result