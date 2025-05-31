import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jnp_zeros

def LineAnalyticalMSD(D, L, t_array, n_terms=1000):
    """
    Compute the analytical MSD for diffusion in a confined line.
    
    Parameters:
        D (float): Diffusion coefficient
        L (float): Length of the Line
        t_array (array-like): Array of time values
        n_terms (int): Number of terms in the series expansion (default: 100)
    
    Returns:
        np.ndarray: MSD values corresponding to t_array
    """
    p_range = list(range(0, n_terms))  # Roots of derivative of Bessel function J1
    msd = np.zeros_like(t_array, dtype=np.float64)
    
    for p in p_range:
        term = np.exp(-( D * t_array * (2*p +1)**2 * np.pi**2)/(L**2)) / (2*p +1)**4
        msd += term

    msd = L**2 * 1/6 * (1 - 96/(np.pi**4) * msd)
    return msd

def CircleAnalyticalMSD(D, a, t_array, n_terms=100):
    """
    Compute the analytical MSD for diffusion in a confined circle.
    
    Parameters:
        D (float): Diffusion coefficient
        a (float): Radius of the circle
        t_array (array-like): Array of time values
        n_terms (int): Number of terms in the series expansion (default: 100)
    
    Returns:
        np.ndarray: MSD values corresponding to t_array
    """
    z_roots = jnp_zeros(1, n_terms)  # Roots of derivative of Bessel function J1
    msd = np.zeros_like(t_array, dtype=np.float64)
    
    for z in z_roots:
        term = np.exp(-z**2 * D * t_array / a**2) / (z**2 * (z**2 - 1))
        msd += term

    msd = a**2 * (1 - 8 * msd)
    return msd

def CylinderAnalyticalMSD(D, a, L, t_array, M=100, P=100):
    """
    Compute the MSD for confined diffusion inside a cylinder.

    Parameters:
        t_array : array_like
            Array of time points.
        a : float
            Cylinder radius.
        L : float
            Cylinder length.
        D : float
            Diffusion coefficient.
        M : int
            Number of Bessel term summations (m).
        P : int
            Number of axial term summations (p).

    Returns:
        msd_array : array_like
            MSD values for each time in t_array.
    """
    alpha_1m = jnp_zeros(1, M)  # Zeros of derivative of J1

    msd = np.zeros_like(t_array, dtype=np.float64)
    const_term = a**2 + L**2 / 6.0                  # constant term is the linear and circlar limit together.

    for m in list(range(1,M)):
        alpha = alpha_1m[m]
        for p in list(range(0,P)):
            lambda_mp = D * ((alpha**2 / a**2) + (np.pi**2 * (2*p + 1)**2 / L**2))
            exp_term = np.exp(-lambda_mp * t_array)

            coeff1 = - 8 * np.pi * L * a**4 / (alpha**2 * (alpha**2 - 1))
            coeff2 = - (np.pi * a**2 * 16 * L**3) / (np.pi**4 * (2*p + 1)**4)


            total_coeff = coeff1 + coeff2
            msd +=  exp_term * total_coeff

    return const_term + (1 / (np.pi * a**2 * L)) * msd


def CylinderAnalyticalMSDGPT(D, a, L, t_array, M=100, P=100):
    """
    Compute the MSD for confined diffusion inside a cylinder.
    
    Parameters:
        D : float
            Diffusion coefficient.
        a : float
            Radius of the cylinder (for radial diffusion).
        L : float
            Length of the cylinder (for axial diffusion).
        t_array : array_like
            Array of time points.
        M : int
            Number of Bessel term summations for radial (m).
        P : int
            Number of axial term summations (p).
    
    Returns:
        msd_array : array_like
            MSD values for each time in t_array.
    """
    # Radial part (confined circle in r–θ plane)
    alpha_1m = jnp_zeros(1, M)  # Zeros of derivative of J1
    msd_radial = np.zeros_like(t_array, dtype=np.float64)
    
    for alpha in alpha_1m:
        term = np.exp(-D * (alpha**2) * t_array / a**2) / (alpha**2 * (alpha**2 - 1))
        msd_radial += term
    
    msd_radial = a**2 * (1 - 8 * msd_radial)  # circle MSD

    # Axial part (1D diffusion between two reflecting walls)
    msd_axial = np.zeros_like(t_array, dtype=np.float64)
    for p in range(P):
        lambda_p = (2 * p + 1) * np.pi / L
        term = np.exp(-D * lambda_p**2 * t_array) / (2 * p + 1)**4
        msd_axial += term

    msd_axial = L**2 / 6 * (1 - (96 / np.pi**4) * msd_axial)  # line MSD

    return msd_radial + msd_axial

