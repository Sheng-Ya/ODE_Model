import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from numba import njit

# outside objective, decide once:
t1_upper_bound = 2.5
t2_upper_bound = 3.8
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
def gaussian_integral(t1, z, B, tau):
    # Compute constants
    mu = t1 + 0.5 * B * tau

    # constant exponent term K
    term1 = - (t1 * t1) / tau
    term2 = (mu ** 2) / tau
    K = term1 + term2

    # erf argument
    arg = (z - mu) / math.sqrt(tau)

    # full integral
    return math.exp(K) * 0.5 * math.sqrt(math.pi * tau) * math.erf(arg)


@njit
def calculate_V_dV_dt(times, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao):
    """
    Updated method for calculating V and dV/dt values
    """
    # Precompute constants
    t1, t2 = initial_guess
    a1, a2, Pt1, Vt1, tau, B = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)

    V = np.zeros(len(times))

    # Breathing cycle patterns
    mask_0_t1 = times <= t1
    mask_t1_t2 = ~mask_0_t1

    x = times[mask_0_t1]
    z = times[mask_t1_t2]

    # Compute constants for solution
    c1 = (Vt1 - ((a1 / E_rs) * t1 + (a2 / E_rs) * (t1 ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t1)) / (
                np.exp(-B * t1) - 1)

    d1 = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) - c1

    # Calculate for 0 <= times <= t1
    V[mask_0_t1] = ((a1 / E_rs) * x - (a1 * R_rs / (E_rs ** 2)) +
                    (a2 / E_rs) * (x ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * x +
                    (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) +
                    c1 * np.exp(-B * x) + d1)

    I0 = gaussian_integral(t1, t1, B, tau)

    I_z = np.array([gaussian_integral(t1, zi, B, tau) for zi in z])
    integral = I_z - I0
    constant = (Vt1 / math.exp(-B * t1)) # - (Pt1 / R_rs) * I0

    V[mask_t1_t2] = (Pt1 / R_rs) * np.exp(-B * z) * integral + constant * np.exp(-B * z)

    return V


@njit
def calculate_P_musc_dP_dt(times, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao, Pmax):
    """
    Updated method for calculating P_musc and dP_musc/dt
    """
    t1, t2 = initial_guess
    a1, a2, Pt1, _, tau, _ = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)

    P_musc = np.zeros(len(times))
    dP_musc_dt = np.zeros(len(times))

    # Breathing cycle patterns
    mask_0_t1 = times <= t1
    mask_t1_t2 = ~mask_0_t1

    # Calculate P_musc for 0 <= times <= t1
    P_musc[mask_0_t1] = a1 * times[mask_0_t1] + a2 * (times[mask_0_t1] ** 2)
    dP_musc_dt[mask_0_t1] = a1 + 2 * a2 * times[mask_0_t1]

    # Calculate P_musc for t1 <= times <= t1 + t2
    P_musc[mask_t1_t2] = Pt1 * np.exp((-(times[mask_t1_t2] - t1) ** 2) / tau)
    # P_musc = np.minimum(P_musc, Pmax)

    dP_musc_dt[mask_t1_t2] = P_musc[mask_t1_t2] * (- 2 * (times[mask_t1_t2] - t1) / tau)

    return P_musc, dP_musc_dt


# @njit
def objective(initial_guess, required_params, VAflow, VD, dt, tolerance):
    """
    Optimized Objective function
    """
    t1, t2 = initial_guess
    # n_steps = int(np.round((t1 + t2) / dt)) + 1
    # times = np.linspace(0, t1 + t2, n_steps)

    T_cycle = t1 + t2
    # Use the *same* base_times for every evaluation
    times = base_times
    # We'll only care about t in [0, T_cycle]
    mask_cycle = times <= T_cycle
    times = times[mask_cycle]

    lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao = required_params

    P_musc, dP_musc_dt = calculate_P_musc_dP_dt(times, initial_guess, VAflow, VD, tolerance, E_rs, R_rs, P_ao, Pmax)
    volume_signal = calculate_V_dV_dt(times, initial_guess, VAflow, VD, tolerance, E_rs, R_rs, P_ao)
    dV_dt_values = (P_musc - P_ao - E_rs * volume_signal) / R_rs

    inspire_index = int(np.round(t1 / dt))
    dV2_dt2_values_squared = ((1 / R_rs) * ((dP_musc_dt - P_ao) -
                                            E_rs * dV_dt_values)) ** 2

    # E1_n = (1 - (P_musc / Pmax)) ** n
    # E2_n = (1 - (np.abs(dP_musc_dt) / Pmax_dot)) ** n

    E1_n = (1 - np.clip((P_musc / Pmax), 0, 0.999999)) ** n
    E2_n = (1 - np.clip((np.abs(dP_musc_dt) / Pmax_dot), 0, 0.999999)) ** n

    # Compute inspiratory and expiratory integrals
    with np.errstate(divide='ignore', invalid='ignore'):
        integrand_inspire = (P_musc[:inspire_index] * dV_dt_values[:inspire_index]) / (
                E1_n[:inspire_index] * E2_n[:inspire_index]) + lambda1 * dV2_dt2_values_squared[:inspire_index]

    integrand_expire = dV2_dt2_values_squared[inspire_index:]

    integral_inspire = simpson(integrand_inspire)
    integral_expire = simpson(integrand_expire)

    plt.plot(volume_signal)
    # plt.plot(dV_dt_values)
    plt.show()  # compare euler to analytical solution

    WI = (1 / (t1 + t2)) * integral_inspire
    WE = (1 / (t1 + t2)) * integral_expire

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

    # Compute constants for solution
    c1 = (Vt1 - ((a1 / E_rs) * t1 + (a2 / E_rs) * (t1 ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t1)) / (
            np.exp(-B * t1) - 1)

    d1 = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) - c1
    c2 = (Vt1 - (Pt1 / R_rs) / (B - 1 / tau)) / np.exp(-B * t1)

    if t <= t1:
        V = ((a1 / E_rs) * t - (a1 * R_rs / (E_rs ** 2)) +
             (a2 / E_rs) * (t ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t +
             (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) +
             c1 * np.exp((-E_rs / R_rs) * t) + d1)
        dV_dt = (1 / R_rs) * (a1 * t + a2 * (t ** 2) - E_rs * V)
    else:
        V = (Pt1 / (R_rs * (B - 1 / tau))) * np.exp((-1 / tau) * (t - t1)) + np.exp(-B * t) * c2
        dV_dt = (1 / R_rs) * (Pt1 * np.exp(-(t - t1) / tau) - E_rs * V)

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
    # V[mask_t1_t2] = math.sqrt(math.pi/a) * math.exp((b ** 2) / (4 * a) - c)
