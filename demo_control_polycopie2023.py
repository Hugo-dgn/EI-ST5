# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm.auto import tqdm
import pickle

# MRG packages
import _env
import preprocessing
import processing
import postprocessing


"""def compute_gradient_descent(chi, grad, domain, mu):
    d = np.roll(grad, 1, axis=1)
    d[:, 0] = 0

    h = np.roll(grad, -1, axis=0)
    h[-1, :] = 0

    b = np.roll(grad, 1, axis=0)
    b[0, :] = 0

    g = np.roll(grad, -1, axis=1)
    g[:, -1] = 0
    
    _grad = d + h + b + g
    
    _chi = chi - mu * _grad
    _chi = processing.set2zero(_chi, domain)
    return _chi"""
    
def BelongsInteriorDomain(node):
	if (node < 0):
		return 1
	if node == 3:
		return 2
	else:
		return 0


def compute_gradient_descent(chi, grad, domain, mu):
	"""This function makes the gradient descent.
	This function has to be used before the 'Projected' function that will project
	the new element onto the admissible space.
	:param chi: density of absorption define everywhere in the domain
	:param grad: parametric gradient associated to the problem
	:param domain: domain of definition of the equations
	:param mu: step of the descent
	:type chi: np.array((M,N), dtype=float64
	:type grad: np.array((M,N), dtype=float64)
	:type domain: np.array((M,N), dtype=int64)
	:type mu: float
	:return chi:
	:rtype chi: np.array((M,N), dtype=float64

	.. warnings also: It is important that the conditions be expressed with an "if",
			not with an "elif", as some points are neighbours to multiple points
			of the Robin frontier.
	"""

	(M, N) = np.shape(domain)
	# for i in range(0, M):
	# 	for j in range(0, N):
	# 		if domain_omega[i, j] != _env.NODE_ROBIN:
	# 			chi[i, j] = chi[i, j] - mu * grad[i, j]
	# # for i in range(0, M):
	# 	for j in range(0, N):
	# 		if preprocessing.is_on_boundary(domain[i , j]) == 'BOUNDARY':
	# 			chi[i,j] = chi[i,j] - mu*grad[i,j]
	# print(domain,'jesuisla')
	#chi[50,:] = chi[50,:] - mu*grad[50,:]
	for i in range(1, M - 1):
		for j in range(1, N - 1):
			#print(i,j)
			#chi[i,j] = chi[i,j] - mu * grad[i,j]
			a = BelongsInteriorDomain(domain[i + 1, j])
			b = BelongsInteriorDomain(domain[i - 1, j])
			c = BelongsInteriorDomain(domain[i, j + 1])
			d = BelongsInteriorDomain(domain[i, j - 1])
			if a == 2:
				chi[i + 1, j] = chi[i + 1, j] - mu * grad[i, j]
			if b == 2:
				chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
			if c == 2:
				chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
			if d == 2:
				chi[i, j - 1] = chi[i, j - 1] - mu * grad[i, j]

	return chi


def compute_projected(chi, domain, V_obj):
    """This function performs the projection of $\chi^n - mu*grad

    To perform the optimization, we use a projected gradient algorithm. This
    function caracterizes the projection of chi onto the admissible space
    (the space of $L^{infty}$ function which volume is equal to $V_{obj}$ and whose
    values are located between 0 and 1).

    :param chi: density matrix
    :param domain: domain of definition of the equations
    :param V_obj: characterizes the volume constraint
    :type chi: np.array((M,N), dtype=float64)
    :type domain: np.array((M,N), dtype=complex128)
    :type float: float
    :return:
    :rtype:
    """

    (M, N) = np.shape(domain)
    S = 0
    for i in range(M):
        for j in range(N):
            if domain[i, j] == _env.NODE_ROBIN:
                S = S + 1

    B = chi.copy()
    l = 0
    chi = processing.set2zero(chi, domain)

    V = np.sum(np.sum(chi)) / S
    debut = -np.max(chi)
    fin = np.max(chi)
    ecart = fin - debut
    # We use dichotomy to find a constant such that chi^{n+1}=max(0,min(chi^{n}+l,1)) is an element of the admissible space
    while ecart > 10 ** -4:
        # calcul du milieu
        l = (debut + fin) / 2
        for i in range(M):
            for j in range(N):
                chi[i, j] = np.maximum(0, np.minimum(B[i, j] + l, 1))
        chi = processing.set2zero(chi, domain)
        V = sum(sum(chi)) / S
        if V > V_obj:
            fin = l
        else:
            debut = l
        ecart = fin - debut
        # print('le volume est', V, 'le volume objectif est', V_obj)

    return chi

def compute_projected2(chi, domain, V_obj):
    chi = processing.set2zero(chi, domain)
    robin = domain == _env.NODE_ROBIN
    S = domain[robin].size
    
    xmin = np.min(chi[robin])
    xmax = np.max(chi[robin])
    x = xmin 
    
    new_chi = chi.copy()
    new_chi[chi < x] = 0
    new_chi[chi >= x] = 1
    
    V = np.sum(np.abs(new_chi[robin]))/S
    
    i = 0
    
    while np.abs(V - V_obj) > 1e-3 and i < 100:
        if V > V_obj:
            xmin = x
            x = (xmax + x)/2
        else:
            xmax = x
            x = (x + xmin)/2
        
        new_chi[chi < x] = 0
        new_chi[chi >= x] = 1
        
        V = np.sum(np.abs(new_chi[robin]))/S
        
        i += 1
    
    return new_chi
        

def optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi.
    """

    k = 0
    (M, N) = np.shape(domain_omega)
    numb_iter = 4
    energy = []
    while k < numb_iter and mu > mu_min:
        u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, chi*Alpha)
        p = processing.solve_helmholtz(domain_omega, spacestep, omega, -2*np.conjugate(u), np.zeros(f_dir.shape), f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, chi*Alpha)
        ene = compute_objective_function(domain_omega, u, spacestep)
        grad = -np.real(Alpha * p * u)
        
        if len(energy) == 0:
            energy.append(ene)
        elif ene < energy[-1]:
            energy.append(ene)
            
        comp_ene = energy[-1]
        
        _chi = None
        while ene >= comp_ene/1.5 and mu > mu_min:
            new_chi = chi.copy()
            new_chi = compute_gradient_descent(new_chi, grad, domain_omega, mu)
            new_chi = compute_projected(new_chi, domain_omega, V_obj)
            new_u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, new_chi*Alpha)
            new_ene = compute_objective_function(domain_omega, new_u, spacestep)
            if new_ene < ene:
                # The step is increased if the energy decreased
                mu = mu / 2
            else:
                # The step is decreased is the energy increased
                mu = mu / 10
            
            if new_ene < energy[-1]:
                energy.append(new_ene)
                _chi = new_chi.copy()
            
            
            ene = new_ene
        if _chi is None:
            break
        else:
            chi = _chi.copy()
        k += 1

    energy = np.array(energy).reshape(-1,1)
    return chi, energy, u, grad

def multi_optimization_procedure(domain_omega, spacestep, list_omega, alpha_OMEGA, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           alpha_ISOREL, mu, chi, V_obj):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi.
    """
    _l = np.abs(alpha_OMEGA - list_omega.reshape(-1, 1))
    indices = np.argmin(_l, axis=1)
    list_alpha = np.array(alpha_ISOREL)[indices]
    k = 0
    (M, N) = np.shape(domain_omega)
    numb_iter = 4
    energy = []
    while k < numb_iter and mu > mu_min:
        list_u = []
        grad = np.zeros(chi.shape)
        
        for omega, Alpha in zip(list_omega, list_alpha):
            u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, chi*Alpha)
            p = processing.solve_helmholtz(domain_omega, spacestep, omega, -2*np.conjugate(u), np.zeros(f_dir.shape), f_neu, f_rob,
                                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, chi*Alpha)
            
            list_u.append(u)
            
            grad += -np.real(Alpha * p * u)
        
        ene = compute_multi_objective_function(domain_omega, list_u, spacestep)
        
        if len(energy) == 0:
            energy.append(ene)
        elif ene < energy[-1]:
            energy.append(ene)
            
        comp_ene = energy[-1]
        
        
        _chi = None
        while ene >= comp_ene/1.5 and mu > mu_min:
            print(ene)
            new_chi = chi.copy()
            new_chi = compute_gradient_descent(new_chi, grad, domain_omega, mu)
            new_chi = compute_projected(new_chi, domain_omega, V_obj)
            
            list_new_u = []
            for omega, Alpha in zip(list_omega, list_alpha):
                new_u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, new_chi*Alpha)
                list_new_u.append(new_u)
                
            new_ene = compute_multi_objective_function(domain_omega, list_new_u, spacestep)
            if new_ene < ene:
                # The step is increased if the energy decreased
                mu = mu / 2
            else:
                # The step is decreased is the energy increased
                mu = mu / 10
            
            if new_ene < energy[-1]:
                energy.append(new_ene)
                _chi = new_chi.copy()
            
            
            ene = new_ene
        if _chi is None:
            break
        else:
            chi = _chi.copy()
        k += 1

    energy = np.array(energy).reshape(-1,1)
    return chi, energy, list_u, grad

def compute_objective_function(domain_omega, u, spacestep):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation.
    """

    energy = np.sum(np.abs(u)**2) * spacestep**2

    return energy

def compute_multi_objective_function(domain_omega, list_u, spacestep):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation.
    """
    energy = 0
    for u in list_u:
        energy += compute_objective_function(domain_omega, u, spacestep)

    return energy


if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level =  0# level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = np.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 2*np.pi*159/340

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_even_chi(M, N, x, y)
    #chi = preprocessing._set_random_chi(M, N, x, y)
    
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    Alpha = 30.0 - 30.0 * 1j
    # -- this is the function you have written during your project
    #import compute_alpha
    #Alpha = compute_alpha.compute_alpha(...)
    alpha_rob = Alpha * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = np.sum(np.sum(chi)) / S  # constraint on the density
    V_obj = 0.4
    
    #chi = chi/np.sum(np.abs(chi))*V_obj
    
    #chi = V_obj*chi
    
    mu = 1e-2  # initial gradient step
    #mu_min = 1e-5  # minimal gradient step
    mu_min = 1e-10 # minimal gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    
    _chi = np.ones(chi.shape)
    _chi = processing.set2zero(_chi, domain_omega)
    
    _chi_rand = preprocessing._set_random_chi(M, N, x, y)
    _chi_rand = processing.set2zero(_chi_rand, domain_omega)
    
    _E_isorel = []
    _E_liner = []
    _E_aero = []
    _E_grad = []
    _E_start = []
    _E_rand = []
    _E_multi_isorel = []
    _E_multi_liner = []
    _E_multi_aero = []
    skip = 1
    
    with open('optim_alpha_g1_config4.pkl', 'rb') as file:
        alpha_OMEGA, alpha_ISOREL = pickle.load(file)
        
    with open('optim_alpha_g1_config1.pkl', 'rb') as file:
        alpha_OMEGA, alpha_LINER = pickle.load(file)
        
    with open('optim_alpha_g1_config5.pkl', 'rb') as file:
        alpha_OMEGA, alpha_AERO = pickle.load(file)
     
     
    _liste_omega = 2*np.pi*np.array([177, 348, 524, 688, 860])
    
    _chi_multi_isorel, _energy_multi_isorel, _, _ = multi_optimization_procedure(domain_omega, spacestep, _liste_omega/340, alpha_OMEGA/340, f, f_dir, f_neu, f_rob,
                                                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                                            alpha_ISOREL, mu, chi.copy(), V_obj)
    _chi_multi_liner, _energy_multi_liner, _, _ = multi_optimization_procedure(domain_omega, spacestep, _liste_omega/340, alpha_OMEGA/340, f, f_dir, f_neu, f_rob,
                                                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                                            alpha_LINER, mu, chi.copy(), V_obj)
    
    _chi_multi_aero, _energy_multi_aero, _, _ = multi_optimization_procedure(domain_omega, spacestep, _liste_omega/340, alpha_OMEGA/340, f, f_dir, f_neu, f_rob,
                                                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                                            alpha_AERO, mu, chi.copy(), V_obj)
    
    for _w, _alpha_isorel, _alpha_liner, _alpha_aero in tqdm(zip(alpha_OMEGA[::skip], alpha_ISOREL[::skip], alpha_LINER[::skip], alpha_AERO[::skip]), total=len(alpha_ISOREL[::skip])):
        _k = _w/340
        
        _u_isorel = processing.solve_helmholtz(domain_omega, spacestep, _k, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, _alpha_isorel*_chi.copy())
        _e_isorel = compute_objective_function(domain_omega, _u_isorel, spacestep)
        _E_isorel.append(_e_isorel)
        
        """_u_liner = processing.solve_helmholtz(domain_omega, spacestep, _k, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, _alpha_liner*_chi.copy())
        _e_liner = compute_objective_function(domain_omega, _u_liner, spacestep)
        _E_liner.append(_e_liner)
        
        _u_aero = processing.solve_helmholtz(domain_omega, spacestep, _k, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, _alpha_aero*_chi.copy())
        _e_aero = compute_objective_function(domain_omega, _u_aero, spacestep)
        _E_aero.append(_e_aero)"""
        
        #_u_start = processing.solve_helmholtz(domain_omega, spacestep, _k, f, f_dir, f_neu, f_rob,
        #                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, chi.copy()*_alpha)
        #_e_start = compute_objective_function(domain_omega, _u_start, spacestep)
        #_E_start.append(_e_start)
    
        #chi, energy, u, grad = optimization_procedure(domain_omega, spacestep, _k, f, f_dir, f_neu, f_rob,
        #                                                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
        #                                                  _alpha, mu, chi.copy(), V_obj)
        #_E_grad.append(np.min(energy))
        
        _u_multi_isorel = processing.solve_helmholtz(domain_omega, spacestep, _k, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, _chi_multi_isorel.copy()*_alpha_isorel)
        _e_multi_isorel = compute_objective_function(domain_omega, _u_multi_isorel, spacestep)
        _E_multi_isorel.append(_e_multi_isorel)
        
        _u_multi_liner = processing.solve_helmholtz(domain_omega, spacestep, _k, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, _chi_multi_liner.copy()*_alpha_liner)
        _e_multi_liner = compute_objective_function(domain_omega, _u_multi_liner, spacestep)
        _E_multi_liner.append(_e_multi_liner)
        
        _u_multi_aero = processing.solve_helmholtz(domain_omega, spacestep, _k, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, _chi_multi_aero.copy()*_alpha_aero)
        _e_multi_aero = compute_objective_function(domain_omega, _u_multi_aero, spacestep)
        _E_multi_aero.append(_e_multi_aero)
        
        #_u_rand = processing.solve_helmholtz(domain_omega, spacestep, _k, f, f_dir, f_neu, f_rob,
        #                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, _chi_rand.copy()*_alpha)
        #_e_rand = compute_objective_function(domain_omega, _u_rand, spacestep)
        #_E_rand.append(_e_rand)
        
        
    plt.figure()
    plt.plot(alpha_OMEGA[::skip]/2/np.pi, _E_isorel, label='ISOREL absorbent everywhere')
    #plt.plot(alpha_OMEGA[::skip]/2/np.pi, _E_liner, label='LINER')
    #plt.plot(alpha_OMEGA[::skip]/2/np.pi, _E_aero, label='AERO')
    #plt.plot(alpha_OMEGA[::skip]/2/np.pi, _E_grad, label='grad')
    plt.plot(alpha_OMEGA[::skip]/2/np.pi, _E_multi_isorel, label='ISOREL multi')
    plt.plot(alpha_OMEGA[::skip]/2/np.pi, _E_multi_liner, label='LINER multi')
    plt.plot(alpha_OMEGA[::skip]/2/np.pi, _E_multi_aero, label='AERO multi')
    #plt.plot(alpha_OMEGA[::skip]/2/np.pi, _E_multi1, label='ISOREL')
    #plt.plot(alpha_OMEGA[::skip]/2/np.pi, _E_rand, label='random')
    #plt.plot(alpha_OMEGA[::skip]/2/np.pi, _E_start, label='start point')
    plt.xlabel('frequency')
    plt.ylabel('energy')
    plt.yscale('log')
    plt.legend(loc='upper right')
    
    plt.figure()
    plt.plot(_energy_multi_aero, label='AERO multi')
    plt.plot(_energy_multi_liner, label='LINER multi')
    plt.plot(_energy_multi_isorel, label='ISOREL multi')
    
    plt.legend(loc='upper right')
    
    plt.show()
    
    #plt.savefig('energy_freq.png')

    #chi, energy = _chi_multi, _energy_multi
    
    """l = []
    m = 10
    for V_obj in tqdm(np.linspace(0, 1, 10)):
    
        _, energy, u, grad = optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                                            Alpha, mu, chi.copy(), V_obj)
        l.append(np.min(energy))
    
    plt.plot(np.linspace(0, 1, 10), l)
    plt.show()"""
    
    
    """#chi, energy, u, grad = multi_optimization_procedure(domain_omega, spacestep, list_omega, f, f_dir, f_neu, f_rob,
    #                                                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                                                    Alpha, mu, chi, V_obj)
    #u = u[0]
    
    chin = chi.copy()
    chin = compute_projected2(chin, domain_omega, V_obj)
    un = u.copy()

    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)
    
    print('End.')"""