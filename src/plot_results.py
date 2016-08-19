#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

d_c = np.genfromtxt("coarse.dat")
d_m = np.genfromtxt("medium.dat")
d_f = np.genfromtxt("fine.dat")
d_e = np.genfromtxt("exact.dat")

plt.plot(d_e[1:,0], d_e[1:,1], lw=1.5, label="exact")
plt.plot(d_c[1:,0], d_c[1:,1], "o", label="coarse")
plt.plot(d_m[1:,0], d_m[1:,1], "s", label="medium", markevery=4)
plt.plot(d_f[1:,0], d_f[1:,1], "v", label="fine", markevery=6)
plt.xlabel("x [m]")
plt.ylabel("T [deg C]")
plt.legend(loc=2)

N = np.array([d_c[0,0], d_m[0,0], d_f[0,0]])
L2 = np.array([d_c[0,1], d_m[0,1], d_f[0,1]])

fig2 = plt.figure()
plt.plot(1.0/N, L2, "ok")
plt.xscale("log")
plt.yscale("log")

slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(1.0/N), np.log10(L2))
print("Order of accuracy: {}".format(slope))

dfdm = np.genfromtxt("fdm.dat")
dcvm = np.genfromtxt("cvm.dat")
dsacvm = np.genfromtxt("sacvm.dat")
fig3 = plt.figure()
plt.plot(dfdm[:,0], dfdm[:,1], "-k" , lw=1.5)
plt.plot(dfdm[:,0], dfdm[:,2], "-b" , lw=1.5)
plt.plot(dfdm[:,0], dfdm[:,3], "-r" , lw=1.5)
plt.plot(dcvm[:,0], dcvm[:,1], "--r" , lw=1.5)
plt.plot(dcvm[:,0], dcvm[:,2], "--k" , lw=1.5)
plt.plot(dcvm[:,0], dcvm[:,3], "--b" , lw=1.5)
plt.plot(dsacvm[:,0], dsacvm[:,1], "or" , lw=1.5)
plt.plot(dsacvm[:,0], dsacvm[:,2], "ok" , lw=1.5)
plt.plot(dsacvm[:,0], dsacvm[:,3], "ob" , lw=1.5)
#plt.legend(loc=2)

plt.show()
