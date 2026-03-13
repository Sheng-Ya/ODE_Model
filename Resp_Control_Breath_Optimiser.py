import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from numba import njit

# outside objective, decide once:
t1_upper_bound = 5
t2_upper_bound = 7
T_max = t1_upper_bound + t2_upper_bound
n_steps = int(np.round(T_max / 0.001)) + 1
base_times = np.linspace(0, T_max, n_steps)


@njit
def compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance):
    """
    Compute constants for the respiratory model
    """
    # Precompute key values
    a2 = (-P_ao - E_rs * VA * (t1 + t2) - E_rs * VD) / (t1 ** 2)
    a1 = -2 * a2 * t1
    Pt1 = a1 * t1 + a2 * (t1 ** 2)
    Vt1 = VA * (t1 + t2) + VD
    tau = max((t2 / (-np.log(tolerance * R_rs / Pt1))), 0.001)
    B = E_rs / R_rs

    return a1, a2, Pt1, Vt1, tau, B

@njit
def trapz_uniform(y, dt):
    s = 0.0
    for i in range(1, len(y)):
        s += 0.5 * (y[i] + y[i-1])
    return s * dt


@njit
def gaussian_integral(mu, z, pref, tau):
    # erf argument
    arg = (z - mu) / np.sqrt(tau)

    # full integral
    return pref * math.erf(arg)


@njit
def calculate_variables(times, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao, Pmax, Pmax_dot, n, lambda1, dt):
    """
    Updated method for calculating P_musc and dP_musc/dt
    """
    t1, t2 = initial_guess
    a1, a2, Pt1, Vt1, tau, B = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)

    P_musc = np.empty(len(times))
    dP_musc_dt = np.empty(len(times))

    V = np.empty(len(times))

    # Breathing cycle patterns
    split_idx = np.searchsorted(times, t1, side="right")

    x = times[:split_idx]
    z = times[split_idx:]

    # Calculate P_musc for 0 <= times <= t1
    P_musc[:split_idx] = a1 * x + a2 * (x ** 2)
    dP_musc_dt[:split_idx] = a1 + 2 * a2 * x

    # Calculate P_musc for t1 <= times <= t1 + t2
    P_musc[split_idx:] = Pt1 * np.exp((-(z - t1) ** 2) / tau)
    # P_musc = np.minimum(P_musc, Pmax)

    dP_musc_dt[split_idx:] = P_musc[split_idx:] * (- 2 * (z - t1) / tau)



    # Compute constants for Volume solution
    c1 = (Vt1 - ((a1 / E_rs) * t1 + (a2 / E_rs) * (t1 ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t1)) / (
            np.exp(-B * t1) - 1)

    d1 = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) - c1

    # Calculate for 0 <= times <= t1
    V[:split_idx] = ((a1 / E_rs) * x - (a1 * R_rs / (E_rs ** 2)) +
                     (a2 / E_rs) * (x ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * x +
                     (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) +
                     c1 * np.exp(-B * x) + d1)

    # Compute constants
    mu = t1 + 0.5 * B * tau

    # constant exponent term K
    term1 = - (t1 * t1) / tau
    term2 = (mu ** 2) / tau
    K = term1 + term2
    pref = np.exp(K) * 0.5 * np.sqrt(np.pi * tau)

    I0 = gaussian_integral(mu, t1, pref, tau)

    I_z = np.empty(len(z))
    for i in range(len(z)):
        I_z[i] = gaussian_integral(mu, z[i], pref, tau)

    integral = I_z - I0
    constant = (Vt1 / np.exp(-B * t1))  # - (Pt1 / R_rs) * I0

    expBz = np.exp(-B * z)

    V[split_idx:] = (Pt1 / R_rs) * expBz * integral + constant * expBz

    dV_dt = (P_musc - P_ao - E_rs * V) / R_rs


    # continue calculations
    dV2_dt2_values_squared = ((1 / R_rs) * ((dP_musc_dt - P_ao) - E_rs * dV_dt)) ** 2

    E1_n = (1 - np.clip((P_musc / Pmax), 0, 0.999999)) ** n
    E2_n = (1 - np.clip((np.abs(dP_musc_dt) / Pmax_dot), 0, 0.999999)) ** n

    integrand_inspire = (P_musc[:split_idx] * dV_dt[:split_idx]) / (
    E1_n[:split_idx] * E2_n[:split_idx]) + lambda1 * dV2_dt2_values_squared[:split_idx]

    integrand_expire = dV2_dt2_values_squared[split_idx:]

    integral_inspire = trapz_uniform(integrand_inspire, dt)
    integral_expire = trapz_uniform(integrand_expire, dt)

    WI = (1 / (t1 + t2)) * integral_inspire
    WE = (1 / (t1 + t2)) * integral_expire

    # plt.scatter(times, volume_signal, s=1)
    # plt.show()
    # plt.scatter(times, dV2_dt2_values_squared, s=1)
    # plt.show()
    # plt.scatter(times, dV_dt_values, s=1)
    # plt.show()

    return WI, WE


# @njit
def objective(initial_guess, required_params, VAflow, VD, dt, tolerance):
    """
    Optimized Objective function
    """
    t1, t2 = initial_guess

    # In objective()
    T_cycle = t1 + t2

    # local grid for this candidate (t1, t2)
    n_steps = max(2000, int(np.ceil(T_cycle / dt)) + 1)
    times = np.linspace(0.0, T_cycle, n_steps)
    dt_local = times[1] - times[0]
    s = np.linspace(0.0, 1.0, 2000)
    times = s * T_cycle
    dt_local = times[1] - times[0]

    # # Use base_times up to T_cycle without boolean masks
    # cycle_idx = np.searchsorted(base_times, T_cycle, side="right")
    # times = base_times[:cycle_idx]

    lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao = required_params

    WI, WE = calculate_variables(times, initial_guess, VAflow, VD, tolerance, E_rs, R_rs, P_ao, Pmax, Pmax_dot, n, lambda1, dt_local)

    # Return cost function value
    return WI + lambda2 * WE


@njit
def calculate_single_V_dV_dt(t, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao):
    """
    Updated method for calculating V and dV/dt values
    """
    # Precompute constants
    t1, t2 = initial_guess
    a1, a2, Pt1, Vt1, tau, B = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)
    expBz = np.exp(-B * t)

    if t <= t1:
        c1 = (Vt1 - ((a1 / E_rs) * t1 + (a2 / E_rs) * (t1 ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t1)) / (
                np.exp(-B * t1) - 1)
        d1 = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) - c1

        V = ((a1 / E_rs) * t - (a1 * R_rs / (E_rs ** 2)) +
             (a2 / E_rs) * (t ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t +
             (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) +
             c1 * expBz + d1)
        dV_dt = (1 / R_rs) * (a1 * t + a2 * (t ** 2) - E_rs * V)
    else:
        mu = t1 + 0.5 * B * tau
        term1 = - (t1 * t1) / tau
        term2 = (mu ** 2) / tau
        K = term1 + term2
        pref = np.exp(K) * 0.5 * np.sqrt(np.pi * tau)

        I0 = gaussian_integral(mu, t1, pref, tau)
        I_z = gaussian_integral(mu, t, pref, tau)

        integral = I_z - I0
        constant = (Vt1 / np.exp(-B * t1))
        V = (Pt1 / R_rs) * expBz * integral + constant * expBz

        P_musc = Pt1 * np.exp((-(t - t1) ** 2) / tau)
        dV_dt = (P_musc - P_ao - E_rs * V) / R_rs

    return V, dV_dt




 # # Calculate for t1 <= times <= t1 + t2
    # integral = integrate_z1_to_z1+z2(np.exp((-z ** 2 / tau) + (B + 2 * t1/tau) * z - (t1 ** 2) / tau))
    # constant = Vt1 * np.exp(B * z) - Pt1 / R_rs * integral
    # V[mask_t1_t2] = (Pt1/R_rs) * np.exp(-B * z) * integral + constant * np.exp(-B * z)

    # EXPIRATION USING GAUSSIAN INTEGRAL
    # Precompute z0 = 0 point for definite integral limits
    # a =  1 / tau
    # b = (B + 2 * t1 / tau)
    # c = (t1 ** 2) / tau
    #
    # V[mask_t1_t2] = np.sqrt(np.pi/a) * np.exp((b ** 2) / (4 * a) - c)
