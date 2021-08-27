# MCMC fit of a polynom to a simulated data set
# we will plot the chain with polynomial_fit_plot_chains.py

import numpy as np
import pylab as plt
from matplotlib import cm

def logprob(d, inv_sig, z, zc, param):
    model = 0
    for count, p in enumerate(param):
        model += p * (z-zc)**count
    res = d - model
    return -0.5 * res.dot(inv_sig).dot(res)

def logprior(param, pmin_array, pmax_array):

    def flat_logp(p, pmin, pmax):
        if (p < pmin) or (p > pmax):
            return(-np.inf)
        else:
            return(1)
    
    def norm_logp(p, mean, std):
        return(-0.5 * ((p-mean)/std)**2)
        
    log_prior = 0
    for count, p in enumerate(param):
        log_prior += flat_logp(p, pmin_array[count], pmax_array[count])
    return log_prior

def proposal(last_point, proposal_std):
    prop = np.random.normal(last_point, proposal_std)
    return(prop)

def acceptance(logp, logp_new):

    if logp_new > logp:
        return True
    else:
        accept = np.random.uniform(0, 1)
        return(accept < (np.exp(logp_new - logp)))




#max likelihood
np.random.seed(1)
n_pts = 100
z_max = 10
zc = 5
true_param = [10, 1, 0.5, 0.01]
n_param = len(true_param)

z = np.linspace(0, z_max, n_pts)
d = np.zeros(n_pts)
P = np.zeros((n_pts, n_param))

for k_par in range(n_param):
    d += true_param[k_par] * z ** k_par
    P[:, k_par] = (z - zc) ** k_par

sig = np.ones(n_pts) * 3
d += np.random.randn(n_pts) * np.sqrt(sig)
inv_sig = np.linalg.inv(np.diag(sig))
temp = np.dot(P.T, inv_sig)
cov = np.linalg.inv(np.dot(temp, P))
a = np.dot(np.dot(cov, temp), d)


#mcmc
param_init = [0, 0, 0, 0]
proposal_std =  np.array([0.67, 0.58, 0.13, 0.0088])/5
pmin_array = [0, 0, 0, -1]
pmax_array = [100, 10, 5, 1]

print(logprob(d, inv_sig, z, zc, param_init))
print(logprior(param_init, pmin_array, pmax_array))
print(proposal(param_init, proposal_std))

accepted = []
accep_count = 0
current_point = param_init

n_steps = 300000
accepted_array = np.zeros((n_param + 1, n_steps))

for i in range(n_steps):
    current_like = logprob(d, inv_sig, z, zc, current_point)
    current_prior = logprior(current_point, pmin_array, pmax_array)
    new_point = proposal(current_point, proposal_std)

    new_like = logprob(d, inv_sig, z, zc, new_point)
    new_prior = logprior(new_point, pmin_array, pmax_array)
    
    if (acceptance(current_like + current_prior,
                        new_like + new_prior)):
        current_point = new_point
        accepted_array[:-1, i] = current_point
        accepted_array[-1, i] = new_like

        accep_count += 1

    else:
        accepted_array[:-1, i] = current_point
        accepted_array[-1, i] = current_like


    print(current_point, current_like, -2 * current_like, accep_count)


np.save("accepted_array.npy",accepted_array)


