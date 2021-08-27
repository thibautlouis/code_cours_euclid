# Visualisation of the fit results


import numpy as np
import pylab as plt

chains_array = np.load("accepted_array.npy")
n_params = chains_array.shape[0] - 1
n_steps = chains_array.shape[1]

burn_out_fraction = 20
chains = {}
for i in range(n_params):
    plt.plot(chains_array[i,:])
    plt.xlabel("samples", fontsize=22)
    chains[i] = chains_array[i][int(burn_out_fraction * n_steps / 100):]
    plt.show()


param_max_like = {}
param_max_like[0] = [28.974318474157894, 0.2598292795313896]
param_max_like[1] = [6.898938956326115, 0.1485594426027948]
param_max_like[2] = [0.6359573817030875, 0.022781161412412633]
param_max_like[3] = [0.004011173027899051, 0.008899166645291908]

corr = np.eye(4)
plt.figure(figsize=(15,8))
for i in range(n_params):
    for j in range(i):
        plt.subplot(n_params, n_params, (i+1) + j*n_params )

        plt.hist2d(chains[i], chains[j], bins=200, density=True, cmap= plt.cm.Greys)
        if (j + 1) == i:
            plt.xlabel(r"$b_{%d}$" % i, fontsize = 22)
            plt.ylabel(r"$b_{%d}$" % j, fontsize = 22)
        cov = np.cov(chains[i], chains[j])
        corr[i,j] = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
plt.show()
plt.matshow(corr)
plt.show()


def gaussian(mean, std):
    x = np.linspace(mean - 5 * std, mean + 5 * std, 1000)
    norm = 1 / np.sqrt(2 * np.pi * std**2)
    gauss = norm * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
    return(x, gauss)


plt.figure(figsize=(15,8))
for i in range(n_params):
    plt.subplot(1, n_params, i+1)
    plt.hist(chains[i], bins = 100, density=True, alpha=0.3)
    mean, std = param_max_like[i]
    x,gauss = gaussian(mean, std)
    plt.plot(x, gauss)
    plt.xlabel(r"$b_{%d}$" % i, fontsize = 22)
    
plt.savefig("histo_and_analytic.png")
plt.clf()
plt.close()


plt.figure(figsize=(15,8))
for i in range(n_params):
    plt.subplot(1, n_params, i+1)
    plt.hist(chains[i], bins = 100, density=True, alpha=0.3)

    mean, std = param_max_like[i]
    x,gauss = gaussian(mean, std)
    plt.plot(x, gauss, alpha=0)

    plt.xlabel(r"$b_{%d}$" % i, fontsize = 22)

plt.savefig("histo.png")
plt.clf()
plt.close()

