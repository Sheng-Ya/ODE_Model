import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import interp1d
from numba import njit



@njit
def simulate_euler(times, P_musc, R_rs, P_ao, E_rs):
    V = np.zeros(len(times))
    dV_dt = np.zeros(len(times))
    dt = np.diff(times)

    for i in range(1, len(times)):
        dV_dt[i-1] = (1 / R_rs) * ((P_musc[i-1] - P_ao) - E_rs * V[i - 1])
        V[i] = V[i - 1] + dV_dt[i-1] * dt[i - 1]

    dV_dt[-1] = (1 / R_rs) * ((P_musc[-1] - P_ao) - E_rs * V[-1])

    return V, dV_dt


class BreathOptimiser:
    def __init__(self, params, VAflow, VD, dt):
        self.params = params
        self.VA = VAflow
        self.VD = VD
        self.dt = dt

    def calculate_V_dV_dt(self, times, initial_Nd_guess):
        [tau, t1, t2] = initial_Nd_guess
        params = self.params
        E_rs = params["E_rs"]
        R_rs = params["R_rs"]
        a2 = (-params["P_ao"] - E_rs * self.VA * (t1 + t2) - E_rs * self.VD) / (t1 ** 2)
        a1 = -2 * a2 * t1

        V = np.zeros(len(times))
        dV_dt = np.zeros(len(times))

        # calculate P_musc at each t
        breath = times % (t1 + t2)
        mask_0_t1 = (0 <= breath) & (breath <= t1)
        mask_t1_t2 = (t1 < breath) & (breath <= (t1 + t2))
        # mask_0_t1[-1] = False
        # mask_t1_t2[-1] = True

        x = breath[mask_0_t1]
        z = breath[mask_t1_t2]
        Pt1 = a1 * t1 + a2 * (t1 ** 2)
        Vt1 = self.VA * (t1 + t2) + self.VD

        c1 = (Vt1 - ((a1 / E_rs) * t1 - (a1 * R_rs/(E_rs**2)) + (a2 / E_rs) * (t1**2) - (2 * a2 * R_rs / (E_rs ** 2)) * t1
                        + (2 * a2 * (R_rs**2) / (E_rs**3)) + (a1*R_rs/(E_rs**2)) - (2 * a2 * (R_rs**2) / (E_rs**3)))) / (np.exp(-(E_rs/R_rs)*t1)-1)
        d1 = (a1 * R_rs / (E_rs**2)) - (2 * a2 * (R_rs**2) / (E_rs**3)) - c1
        c2 = (Vt1 - (Pt1 / R_rs) * (1/(E_rs/R_rs - 1/tau))) / np.exp(-(E_rs/R_rs)*t1)
        B = E_rs / R_rs

        # Calculate V for breath in the range 0 to t1
        V[mask_0_t1] = ((a1 / E_rs) * x - (a1 * R_rs/(E_rs**2)) + (a2 / E_rs) * (x**2) - (2 * a2 * R_rs / (E_rs ** 2)) * x
                        + (2 * a2 * (R_rs**2) / (E_rs**3)) + c1 * np.exp((-E_rs/R_rs) * x) + d1)

        dV_dt[mask_0_t1] = (1/R_rs) * (a1 * x  + a2 * (x**2) - E_rs * V[mask_0_t1])

        # Calculate V for breath in the range t1 to t1 + t2
        V[mask_t1_t2] = np.exp(-B*z) * (Pt1 / R_rs) * (1/(B - 1/tau)) * np.exp(((B - 1/tau) * z) + t1/tau) + np.exp(-B*z) * c2
        dV_dt[mask_t1_t2] = (1/R_rs) * (Pt1 * np.exp(-(z-t1)/tau) - E_rs * V[mask_t1_t2])

        return V, dV_dt

    # def calculate(self, times, initial_Nd_guess):
    #     [tau, t1, t2] = initial_Nd_guess
    #     params = self.params
    #     E_rs = params["E_rs"]
    #     R_rs = params["R_rs"]
    #     a2 = (-params["P_ao"] - E_rs * self.VA * (t1 + t2) - E_rs * self.VD) / (t1 ** 2)
    #     a1 = -2 * a2 * t1
    #
    #     V = np.zeros(len(times))
    #     dV_dt = np.zeros(len(times))
    #
    #     # calculate P_musc at each t
    #     breath = times % (t1 + t2)
    #     mask_0_t1 = (0 <= breath) & (breath <= t1)
    #     mask_t1_t2 = (t1 < breath) & (breath <= (t1 + t2))
    #     mask_0_t1[-1] = True
    #     mask_t1_t2[-1] = False
    #
    #     x = breath[mask_0_t1]
    #     z = breath[mask_t1_t2]
    #     Pt1 = a1 * t1 + a2 * (t1 ** 2)
    #     Vt1 = self.VA * (t1 + t2) + self.VD
    #
    #     d1 = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3))
    #     c1 = (Vt1 - ((a1 / E_rs) * t1 - (a1 * R_rs / (E_rs ** 2)) + (a2 / E_rs) * (t1 ** 2) - (
    #                 2 * a2 * R_rs / (E_rs ** 2)) * t1
    #                  + (2 * a2 * (R_rs ** 2) / (E_rs ** 3)))) / np.exp(-(E_rs / R_rs) * t1)
    #     c2 = (Vt1 - (Pt1 / R_rs) * (1 / (E_rs / R_rs - 1 / tau))) / np.exp(-(E_rs / R_rs) * t1)
    #     B = E_rs / R_rs
    #
    #     # Calculate V for breath in the range 0 to t1
    #     V[mask_0_t1] = ((a1 / E_rs) * x - (a1 * R_rs / (E_rs ** 2)) + (a2 / E_rs) * (x ** 2) - (
    #                 2 * a2 * R_rs / (E_rs ** 2)) * x
    #                     + (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) + c1 * np.exp((-E_rs / R_rs) * x))
    #
    #     check_3 = ((a1 / E_rs) * t1 - (a1 * R_rs / (E_rs ** 2)) + (a2 / E_rs) * (t1 ** 2) - (
    #                 2 * a2 * R_rs / (E_rs ** 2)) * t1
    #                + (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) + d1 * np.exp((-E_rs / R_rs) * t1))
    #
    #     dV_dt[mask_0_t1] = (1 / R_rs) * (a1 * x + a2 * (x ** 2) - E_rs * V[mask_0_t1])
    #
    #     # Calculate V for breath in the range t1 to t1 + t2
    #     V[mask_t1_t2] = np.exp(-B * z) * (Pt1 / R_rs) * (1 / (B - 1 / tau)) * np.exp(
    #         ((B - 1 / tau) * z) + t1 / tau) + np.exp(-B * z) * c2
    #     dV_dt[mask_t1_t2] = (1 / R_rs) * (Pt1 * np.exp(-(z - t1) / tau) - E_rs * V[mask_t1_t2])
    #
    #     return V, dV_dt

    def tau_constraint(self, initial_Nd_guess):
        [tau, t1, t2] = initial_Nd_guess
        params = self.params

        a2 = (-params["P_ao"] - params["E_rs"] * self.VA * (t1 + t2) - params["E_rs"] * self.VD) / (t1 ** 2)
        a1 = -2 * a2 * t1

        Pt1 = a1 * t1 + a2 * (t1 ** 2)
        constraint = Pt1 * np.exp(-t2 / tau)
        return constraint


    def calculate_P_musc_dP_dt(self, times, initial_Nd_guess):
        [tau, t1, t2] = initial_Nd_guess
        params = self.params

        a2 = (-params["P_ao"] - params["E_rs"] * self.VA * (t1 + t2) - params["E_rs"] * self.VD) / (t1 ** 2)
        a1 = -2 * a2 * t1

        P_musc = np.zeros(len(times))
        dP_musc_dt = np.zeros(len(times))
        P_musc_t1 = a1 * t1 + a2 * (t1 ** 2)

        # calculate P_musc at each t
        breath = times % (t1 + t2)
        mask_0_t1 = (0 <= breath) & (breath <= t1)
        mask_t1_t2 = (t1 < breath) & (breath <= (t1 + t2))
        # mask_0_t1[-1] = False
        # mask_t1_t2[-1] = True

        # Calculate P_musc for breath in the range 0 to t1
        P_musc[mask_0_t1] = a1 * breath[mask_0_t1] + a2 * (breath[mask_0_t1] ** 2)
        dP_musc_dt[mask_0_t1] = a1 + 2 * a2 * breath[mask_0_t1]

        # Calculate P_musc for breath in the range t1 to t1 + t2
        P_musc[mask_t1_t2] = P_musc_t1 * np.exp(-(breath[mask_t1_t2] - t1) / tau)
        dP_musc_dt[mask_t1_t2] = P_musc_t1 * np.exp(-(breath[mask_t1_t2] - t1) / tau) * (-1 / tau)

        return P_musc, dP_musc_dt


    def objective(self, initial_Nd_guess):
        """
         Function to obtain a0, a1, a2, tau, t1, t2 Edit: removed a0 and a1
        """
        [tau, t1, t2] = initial_Nd_guess
        params = self.params

        n_steps = int(np.round((t1 + t2) / self.dt)) + 1
        times = np.linspace(0, (t1 + t2), n_steps)

        # Breathing Pattern Optimiser
        lambda1 = params["lambda1"]
        lambda2 = params["lambda2"]
        n = params["n"]
        Pmax = params["Pmax"]
        Pmax_dot = params["Pmax_dot"]

        P_musc, dP_musc_dt = self.calculate_P_musc_dP_dt(times, initial_Nd_guess)

        # volume_signal, dV_dt_values  = simulate_euler(times, P_musc, params["R_rs"], params["P_ao"], params["E_rs"])

        volume_signal, dV_dt_values = self.calculate_V_dV_dt(times, initial_Nd_guess)

        # print((-2 * a2 * t1), a2, tau, t1, t2)
        # plt.plot(volume_signal)
        # plt.plot(check_V)
        # plt.show()

        inspire_index = int(round(t1 / self.dt))
        dV2_dt2_values_squared = ((1 / params["R_rs"]) * ((dP_musc_dt - params["P_ao"]) - params["E_rs"] * dV_dt_values)) ** 2

        E1_n = (1 - P_musc / Pmax) ** n
        E2_n = (1 - dP_musc_dt / Pmax_dot) ** n

        integrand_inspire = (P_musc[:inspire_index] * dV_dt_values[:inspire_index]) / (E1_n[:inspire_index] * E2_n[:inspire_index]) + lambda1 * dV2_dt2_values_squared[:inspire_index]
        integrand_expire = dV2_dt2_values_squared[inspire_index:]

        # Integrate over time using Simpson’s rule
        integral_inspire = simpson(integrand_inspire, x=times[:inspire_index])
        integral_expire = simpson(integrand_expire, x=times[inspire_index:])

        WI = (1 / (t1 + t2)) * integral_inspire
        WE = (1 / (t1 + t2)) * integral_expire

        J = WI + lambda2 * WE

        return J

    # didn't do ln. Original paper has ln, but the Carlos paper does not