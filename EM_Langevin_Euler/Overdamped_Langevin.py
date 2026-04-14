'''
Elodie Millan
June 2020
(Update 2023)
---------
Numerical simulation of Bulk Brownian motion.
'''

import numpy as np
import time

class Langevin:
    def __init__(self, dt, Nt, a, eta0=0.001, T=300, X0=0, gravity=False, signe="+"):
        """
        Constructor : Initialisation.

        :param dt: Numerical time step [s].
        :param Nt: Number of point.
        :param a: Particle radius [m].
        :param eta0: Fluid viscosity (default = 0.001 [Pa/s]).
        :param T: Temperature (default = 300 [K]).
        :param X0: Initial position (default = 0 [m]).
        :param gravity : Boolean to activate gravity (default = False).
        :param signe : String "+" or "-" to indicate orientation of gravity (Default = "+").
        """
        self.dt = dt
        self.Nt = int(Nt)
        self.a = a
        self.eta0 = eta0
        self.T = T
        self.Xn = np.zeros(Nt)
        self.Xn[0] = X0
        self.gravity = gravity
        if signe=="+":
            self.pm = +1
        else:
            self.pm = -1

        self.lB = 526e-9
        self.kb = 1.38e-23
        self.gamma = 6 * np.pi * self.eta0 * self.a
        self.D0 = (self.kb * self.T) / (self.gamma)
        self.t = np.arange(Nt) * dt

    def trajectory(self, output=False):
        """
        @param output: Boolean, if True return Xn (default = False).
        @return: Trajectory Xn.
        """
        for n in range(self.Nt-1):
            Wn = random_gaussian()
            if self.gravity:
                self.U = self.pm*(self.kb*self.T)/self.lB / self.gamma # Signe of speed from force given by lB in input.
            else:
                self.U = 0
            self.Xn[n+1] = self.Xn[n] + self.U*self.dt + np.sqrt(2*self.D0)* Wn * np.sqrt(self.dt)
        if output:
            return self.Xn

    '''
    Méthodes pour l'analyse
    '''
    def MSD(self):
        """
        @return: (< [Xn(t+tau) - Xn(t)]^2 >, tau).
        """
        self.list_k_tau = np.array([], dtype=int) # Liste des entiers k tel que tau = k*dt
        for k in range(len(str(self.Nt)) - 1):
            # Construit 10 points par décades
            self.list_k_tau = np.concatenate(
                (
                    self.list_k_tau,
                    np.arange(10 ** k, 10 ** (k + 1), 10 ** k, dtype=int),
                )
            )

        x = self.Xn
        NumberOfMSDPoint = len(self.list_k_tau)
        self.MSD = np.zeros(NumberOfMSDPoint)
        for n, k in enumerate(self.list_k_tau):
            if k == 0:
                self.MSD[n] = 0
                continue
            self.MSD[n] = np.mean((x[k:] - x[0:-k]) ** 2)

        return self.list_k_tau*self.dt, self.MSD

'''
MERSENNE-TWISTER ALGORiTHM IN PYTHON
Copyright (c) 2019 yinengy
'''
# Coefficients for MT19937
(w, n, m, r) = (32, 624, 397, 31)
a = 0x9908B0DF
(u, d) = (11, 0xFFFFFFFF)
(s, b) = (7, 0x9D2C5680)
(t, c) = (15, 0xEFC60000)
l = 18
f = 1812433253

# Make a arry to store the state of the generator
MT = [0 for i in range(n)]
index = n+1
lower_mask = 0x7FFFFFFF #(1 << r) - 1 // le nombre binaire de r
upper_mask = 0x80000000 # w bits les plus bas de (pas lower_mask)

# Initialize the generator from a seed
def mt_seed(seed):
    # global index
    # index = n
    MT[0] = seed
    for i in range(1, n):
        temp = f * (MT[i-1] ^ (MT[i-1] >> (w-2))) + i
        MT[i] = temp & 0xffffffff

# Extract a tempered value based on MT[index]
# Calling twist() every n numbers
def random_mersenne_twister():
    global index
    if index >= n:
        twist()
        index = 0

    y = MT[index]
    y = y ^ ((y >> u) & d)
    y = y ^ ((y << s) & b)
    y = y ^ ((y << t) & c)
    y = y ^ (y >> l)

    index += 1
    return (y & 0xffffffff)/4294967296

# Generate the next n values from the series x_i
def twist():
    for i in range(0, n):
        x = (MT[i] & upper_mask) + (MT[(i+1) % n] & lower_mask)
        xA = x >> 1
        if (x % 2) != 0:
            xA = xA ^ a
        MT[i] = MT[(i + m) % n] ^ xA

'''
BOX-MULLER ALGORITHM IN PYTHON
'''
def random_gaussian():
    """
    :return: Gaussian random number of 0 mean and 1 standard deviation.
    """
    w = 2.0
    while (w >= 1.0):
        # genère une seed sur le temp en 10^-7 second
        mt_seed(int(time.time()*1e7))
        x1 = 2.0 * random_mersenne_twister() - 1.0
        x2 = 2.0 * random_mersenne_twister() - 1.0
        w = x1 * x1 + x2 * x2
    w = ((-2.0 * np.log(w)) / w) ** 0.5
    return x1 * w



# ##############################################################
# ### DEBBUG ZONE WITH SOME TEST()
# ##############################################################
# def test():
#     import matplotlib.pyplot as plt
#     from matplotlib import rc
#     import seaborn as sns
#     custom_params = {
#         "xtick.direction": "in",
#         "ytick.direction": "in",
#         "lines.markeredgecolor": "k",
#         "lines.markeredgewidth": 0.3,
#         "figure.dpi": 200,
#         "text.usetex": True,
#         "font.family": "serif",
#     }
#     sns.set_theme(context="paper", style="ticks", rc=custom_params)
#     '''
#     Checking the distribution of the random_gaussian() function
#     '''
#     # sigma = 1
#     # N = 10000
#     # Test_gauss = np.zeros(N)
#     # for i in range(len(Test_gauss)):
#     #     Test_gauss[i] = random_gaussian() * sigma
#     # # Distribution de Test_gauss
#     # plt.figure(figsize = (1.5*3.375, 1.5*3.375/1.68),  tight_layout=True)
#     # plt.hist(Test_gauss, bins=50, density=True)
#     # x = np.linspace(-3 * sigma, 3 * sigma, 1000)
#     # p_gauss = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-x ** 2 / (2 * sigma ** 2))
#     # plt.plot(x, p_gauss / np.trapz(p_gauss, x), "k-")
#     # titre = r"Probabilité de $" + str(N) + r"$ tirages de random_gaussian() de variance $\sigma = " + str(sigma) + "$"
#     # plt.title(titre)
#     # plt.xlabel(r"$X \sim \mathcal{N}(0,1)$")
#     # plt.ylabel(r"$P(X)$")
#     # plt.show()
#     '''
#     Overdamped_Langevin_Python debugging routine
#     '''
#     langevin = Langevin(dt=0.01, Nt=100_000, a=1.5e-6)
#     langevin.trajectory()

#     ###Plot MSD
#     delta_t, MSD = langevin.MSD()
#     dt_theo = np.linspace(langevin.dt, langevin.dt*langevin.Nt, 100)
#     plt.figure(figsize=(1.5 * 3.375, 1.5 * 3.375 / 1.68), tight_layout=True)
#     plt.loglog(delta_t, MSD, "o")
#     plt.plot(dt_theo, 2*langevin.D0*dt_theo, "k-")
#     plt.xlabel(r"$t$")
#     plt.ylabel(r"$\langle X_t^2 \rangle$")
#     plt.show()

#     ###Plot distribution
#     N_tau = 10
#     sigma = np.sqrt(2*langevin.D0*N_tau*langevin.dt)
#     delta_Xnau = langevin.Xn[N_tau:]-langevin.Xn[:-N_tau]
#     fig = plt.figure(figsize=(1.5 * 3.375, 1.5 * 3.375 / 1.68), tight_layout=True)
#     plt.hist(delta_Xnau, bins=50, density=True)
#     x = np.linspace(-3 * sigma, 3 * sigma, 1000)
#     p_gauss = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-x ** 2 / (2 * sigma ** 2))
#     plt.plot(x, p_gauss / np.trapz(p_gauss, x), "k-")
#     plt.xlabel(r"$\Delta X_t \sim \mathcal{N}(0, \sqrt{2 D_0 \tau} )$")
#     plt.ylabel(r"$P(\Delta X_{t=\tau})$")
#     plt.show()


# if __name__ == '__main__':
#     test()