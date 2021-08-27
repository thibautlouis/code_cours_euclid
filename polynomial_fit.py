# Analytic fit of a polynom to a simulated data set
import numpy as np
import pylab as plt
from matplotlib import cm

np.random.seed(1)

n_pts = 100
z_max = 10
param = [10, 1, 0.5, 0.01]
n_param = len(param)
zc = 5

z = np.linspace(0, z_max, n_pts)
d = np.zeros(n_pts)
P = np.zeros((n_pts, n_param))

for par in range(n_param):
    d += param[par] * z ** par
    P[:, par] = (z-zc) ** par

sig = np.ones(n_pts) * 3
d += np.random.randn(n_pts) * np.sqrt(sig)

plt.errorbar(z-zc, d, np.sqrt(sig), fmt=".")
plt.xlabel("z-zc",fontsize=16)
plt.ylabel("data",fontsize=16)
plt.show()


inv_sig = np.linalg.inv(np.diag(sig))
temp = np.dot(P.T, inv_sig)
cov = np.linalg.inv(np.dot(temp, P))

a = np.dot(np.dot(cov, temp), d)

print(param)
print(a)
print(cov)

model = np.zeros(n_pts)
for par in range(n_param):
    model += a[par] * (z-zc) ** par
    print(par, a[par], np.sqrt(cov[par, par]))


plt.errorbar(z - zc, d, np.sqrt(sig), fmt=".")
plt.errorbar(z - zc, model, color="red", label="F(z)")
plt.xlabel(r"z-zc",fontsize=16)
plt.ylabel("data",fontsize=16)
plt.legend(fontsize=16)
plt.show()


plt.errorbar(z - zc, d, np.sqrt(sig), fmt=".")
plt.errorbar(z - zc, a[0] * (z - zc) ** 0, color="green", label=r"$b_{0}$")
plt.errorbar(z - zc, a[1] * (z - zc) ** 1, color="brown", label=r"$b_{1}(z-zc)$")
plt.errorbar(z - zc, a[2] * (z - zc) ** 2, color="orange", label=r"$b_{2}(z-zc)^{2}$")
plt.errorbar(z - zc, a[3] * (z - zc) ** 3, color="purple", label=r"$b_{3}(z-zc)^{3}$")
plt.errorbar(z - zc, model, color="red", label="F(z)")
plt.xlabel(r"z-zc",fontsize=16)
plt.ylabel("data",fontsize=16)
plt.legend(fontsize=16)
plt.show()
