import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import interp1d

class BreathOptimiser:
    def __init__(self, params, times, dt, overall_times):
        self.params = params
        self.times = times
        self.dt = dt
        self.overall_times = overall_times
        # self.V_t1 = None # Store the volume at the t1 index

    def constraint_function(self, initial_Nd_guess, VD, VA):
        [a2, tau, t1, t2] = initial_Nd_guess
        params = self.params
        a1 = -2 * a2 * t1
        constraint = (a1 * t1 + a2 * (t1 **2) - params["P_ao"]) - params["E_rs"] * (VA * (t1 + t2) + VD)
        return constraint

    def tau_constraint(self, initial_Nd_guess):
        [a2, tau, t1, t2] = initial_Nd_guess
        a1 = -2 * a2 * t1
        Pt1 = a1 * t1 + a2 * (t1 ** 2)
        constraint = Pt1 * np.exp(-t2 / tau)
        return constraint

    def simulate_euler(self, V0, times, params, P_musc):
        V = np.zeros(len(times))
        dV_dt = np.zeros(len(times))
        V[0] = V0

        for i in range(1, len(times)):
            dV_dt[i] = (1 / params["R_rs"]) * ((P_musc[i - 1] - params["P_ao"]) - params["E_rs"] * V[i - 1])
            V[i] = V[i - 1] + dV_dt[i] * self.dt[i - 1]

        return V, dV_dt

    # def constraint_positive(self, initial_Nd_guess):
    #     # Ensure P_musc is always greater than or equal to zero
    #     [a1, a2, tau, t1, t2] = initial_Nd_guess
    #     times = self.times
    #     P_musc = [0] * len(times)
    #
    #     # Calculate P_musc at each t
    #     for j in range(len(times)):
    #         breath = times[j] % (t1 + t2)
    #         if 0 <= breath <= t1:
    #             P_musc[j] = a1 * breath + a2 * (breath ** 2)
    #         elif t1 < breath <= (t1 + t2):
    #             P_musc_t1 = a1 * t1 + a2 * (t1 ** 2)
    #             P_musc[j] = P_musc_t1 * np.exp(-(breath - t1) / tau)
    #
    #     # Return the difference from zero (to enforce P_musc >= 0)
    #     return min(P_musc)  # The optimizer will try to keep P_musc >= 0


    # def dV_dt_function(self, V, params, P):
    #     G_AW = 1
    #     R_rs = params["R_rs"]
    #     P_ao = params["P_ao"]
    #     E_rs = params["E_rs"]
    #
    #     return (G_AW / R_rs) * ((P - P_ao) - E_rs * V)


    def objective(self, initial_Nd_guess):
        """
         Function to obtain a0, a1, a2, tau, t1, t2 Edit: removed a0 and a1
        """
        [a2, tau, t1, t2] = initial_Nd_guess
        params = self.params
        times = self.times

        a1 = -2 * a2 * t1

        # Breathing Pattern Optimiser
        lambda1 = params["lambda1"]
        lambda2 = params["lambda2"]
        n = params["n"]
        Pmax = params["Pmax"]
        Pmax_dot = params["Pmax_dot"]

        P_musc = np.zeros(len(times))
        dP_musc_dt = np.zeros(len(times))
        P_musc_t1 = a1 * t1 + a2 * (t1 ** 2)

        # calculate P_musc at each t
        breath = times % (t1 + t2)
        mask_0_t1 = (0 <= breath) & (breath <= t1)
        mask_t1_t2 = (t1 < breath) & (breath <= (t1 + t2))

        # Calculate P_musc for breath in the range 0 to t1
        P_musc[mask_0_t1] = a1 * breath[mask_0_t1] + a2 * (breath[mask_0_t1] ** 2)
        dP_musc_dt[mask_0_t1] = a1 + 2 * a2 * breath[mask_0_t1]

        # Calculate P_musc for breath in the range t1 to t1 + t2
        P_musc[mask_t1_t2] = P_musc_t1 * np.exp(-(breath[mask_t1_t2] - t1) / tau)
        dP_musc_dt[mask_t1_t2] = P_musc_t1 * np.exp(-(breath[mask_t1_t2] - t1) / tau) * (-1 / tau)

        volume_signal, dV_dt_values  = self.simulate_euler(0.0, times, self.params, P_musc)

        inspire_index = np.argmin(np.abs((times % (t1 + t2)) - t1))
        dV2_dt2_values_squared = ((1 / params["R_rs"]) * ((dP_musc_dt - params["P_ao"]) - params["E_rs"] * dV_dt_values)) ** 2

        E1_n = (1 - P_musc / Pmax) ** n
        E2_n = (1 - dP_musc_dt / Pmax_dot) ** n

        integrand_inspire = (P_musc[:inspire_index] * dV_dt_values[:inspire_index]) / (E1_n[:inspire_index] * E2_n[:inspire_index]) + lambda1 * dV2_dt2_values_squared[:inspire_index]
        integrand_expire = dV2_dt2_values_squared[inspire_index:]

        # print(f"integrand_inspire.shape: {integrand_inspire.shape}")
        # print(f"times[:inspire_index].shape: {times[:inspire_index].shape}")
        # print(f"integrand_expire.shape: {integrand_expire.shape}")
        # print(f"times[inspire_index:].shape: {times[inspire_index:].shape}")

        # A = self.overall_times[-1]

        if self.overall_times[-1] > 3:
            print(f"guess: {initial_Nd_guess}")
            plt.plot(times, volume_signal, marker='o')
            plt.plot(times, dV_dt_values, marker='o')
            plt.show()

        # Integrate over time using Simpson’s rule
        integral_inspire = simpson(integrand_inspire, x=times[:inspire_index])
        integral_expire = simpson(integrand_expire, x=times[inspire_index:])

        WI = (1 / (t1 + t2)) * integral_inspire
        WE = (1 / (t1 + t2)) * integral_expire

        # A = volume_signal[inspire_index]
        # self.V_t1 = A

        J = WI + lambda2 * WE

        return J












    # # other inputs
    # dV_dt = updates["dV_dt_history"]
    # P_musc = updates["P_musc_history"]
    # times = updates["time_breath_history"]
    # dP_musc_dt = [0] * len(times)
    # d2V_dt2_squared = [0] * len(times)
    # E1_n = [0] * len(times)
    # E2_n = [0] * len(times)
    #
    # for j in range(len(times)):
    #     if times[j] == 0:
    #         dP_musc_dt[0] = a1
    #         d2V_dt2_squared[0] = 0
    #         E1_n[0] = 1
    #         E2_n[0] = 1
    #     else:
    #         breath = times[j] % (t1 + t2)
    #         if 0 <= breath <= t1:
    #             dP_musc_dt[j] = a1 + 2 * a2 * times[j]
    #             E1_n[j] = (1 - P_musc[j] / Pmax) ** n
    #             E2_n[j] = (1 - dP_musc_dt[j] / Pmax_dot) ** n
    #
    #         # elif t1 < breath <= (t1 + t2):
    #         #     P_musc_t1 = a0 + a1 * t1 + a2 * (t1 ** 2)
    #         #     dP_musc_dt[j] = P_musc_t1 * np.exp(-(times[j] - t1) / tau) * (-1 / tau)
    #
    #         if j == 0:
    #             d2V_dt2_squared[j] = ((dV_dt[j+1] - dV_dt[j]) / (times[j+1] - times[j])) ** 2
    #         else:
    #             d2V_dt2_squared[j] = ((dV_dt[j] - dV_dt[j-1]) / (times[j] - times[j-1])) ** 2
    #
    # # Compute the integrand at each time point
    # time_in_breath = times % (t1 + t2)
    # inspire_index = np.where(time_in_breath > t1)[0]
    #
    # integrand_inspire = P_musc[:inspire_index] * dV_dt[:inspire_index] / (E1_n[:inspire_index] * E1_n[:inspire_index]) + lambda1 * (dV_dt[:inspire_index] ** 2)
    # integrand_expire = d2V_dt2_squared[inspire_index:]
    #
    #
    # # Integrate over time using Simpson’s rule
    # integral_inspire = simpson(integrand_inspire, times[:inspire_index])
    # integral_expire = simpson(integrand_expire, times[inspire_index:])
    #




    # previous_WI = exp_inputs["previous_WI"][i - 1]
    # previous_WE = exp_inputs["previous_WE"][i - 1]
    #
    #
    # if t == 0:
    #     dP_musc_dt = a1
    #     d2V_dt2_squared = 0
    # else:
    #     breath = t % (t1 + t2)
    #     if 0 <= breath <= t1:
    #         dP_musc_dt = a1 + 2 * a2 * t
    #     elif t1 < breath <= (t1 + t2):
    #         P_musc_t1 = a0 + a1 * t1 + a2 * (t1 ** 2)
    #         dP_musc_dt = P_musc_t1 * np.exp(-(t - t1) / tau) * (-1 / tau)
    #
    #     d2V_dt2_squared = ((previous_dV_dt - dV_dt) / step_size) ** 2
    #
    # E1_n = (1 - P_musc / Pmax) ** n
    # E2_n = (1 - dP_musc_dt / Pmax_dot) ** n
    #
    # if 0 <= breath <= t1:
    #     dWI_dt = (1/(t1+t2)) * (P_musc * dV_dt / (E1_n * E2_n)) + lambda1 * d2V_dt2_squared
    #     WI = previous_WI + dWI_dt * step_size  # Integrate using Euler's method
    # else:
    #     dWE_dt = (1/(t1+t2)) * d2V_dt2_squared
    #     WE = previous_WE + dWE_dt * step_size
    #
    #
    # previous_WI = WI
    # previous_WE = WE

    # J = WI + lambda2 * WE
    #
    # # t_eval = updates["t_eval4"][0]
    # # tolerance = 1e-3
    # # if np.abs(t - t_eval) < tolerance:
    #     # Store WI and WE globally
    # updates["WI"].append(WI)
    # updates["WE"].append(WE)
    # updates["previous_WI"].append(previous_WI)
    # updates["previous_WE"].append(previous_WE)
    #
    # # updates["t_eval4"] = updates["t_eval4"][1:]
    #
    # # time_history.append(t)
    #
    # return J



