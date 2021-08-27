import numpy as np
import pylab as plt
from scipy.stats import chi2

size = 70
dof = 50
chi2_array = np.random.chisquare(dof, size=size)


n_test = np.arange(size)

plt.figure(figsize=(12,8))
plt.xlabel("index du test", fontsize=22)
plt.ylabel(r"$\chi^{2}_{\rm null}$", fontsize=22)
plt.plot(n_test, chi2_array)
plt.plot(n_test, n_test*0 +50)
plt.savefig("chi2_lee.png", bbox_inches='tight')
plt.clf()
plt.close()

p = 1 - chi2.cdf(chi2_array, dof)

plt.figure(figsize=(12,8))
plt.xlabel("index du test", fontsize=22)
plt.ylabel(r"$p(\chi^{2} > \chi_{\rm mesure}^{2})$ (%)", fontsize=22)
plt.plot(n_test, p * 100)
plt.savefig("p_lee.png", bbox_inches='tight')
plt.clf()
plt.close()
