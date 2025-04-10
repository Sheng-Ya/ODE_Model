import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import interp1d

class BreathOptimiser:
    def __init__(self, params, times):
        self.params = params
        self.times = times
        self.V_t1 = None # Store the volume at the t1 index

    def constraint_function(self, initial_Nd_guess, VD, VA):
        [_, _, _, t1, t2] = initial_Nd_guess
        V_t1 = self.latest_volume
        return (V_t1 - VD) * (1 / (t1 + t2)) - VA


    def dV_dt_function(self, t, V, params, p_musc_func):
        G_AW = 1
        R_rs = params["R_rs"]
        P_ao = params["P_ao"]
        E_rs = params["E_rs"]

        P = p_musc_func(t)
        return (G_AW / R_rs) * ((P - P_ao) - E_rs * V)


    def objective(self, initial_Nd_guess):
        """
         Function to obtain a0, a1, a2, tau, t1, t2 Edit: removed a0
        """
        [a1, a2, tau, t1, t2] = initial_Nd_guess
        params = self.params
        times = self.times

        # Breathing Pattern Optimiser
        lambda1 = params["lambda1"]
        lambda2 = params["lambda2"]
        n = params["n"]
        Pmax = params["Pmax"]
        Pmax_dot = params["Pmax_dot"]

        P_musc = [0] * len(times)
        dP_musc_dt = [0] * len(times)

        # calculate P_musc at each t
        for j in range(len(times)):
            breath = times[j] % (t1 + t2)
            if 0 <= breath <= t1:
                P_musc[j] = a1 * breath + a2 * (breath ** 2)
                dP_musc_dt[j] = a1 + 2 * a2 * breath
            elif t1 < breath <= (t1 + t2):
                P_musc_t1 = a1 * t1 + a2 * (t1 ** 2)
                P_musc[j] = P_musc_t1 * np.exp(-(breath - t1) / tau)
                dP_musc_dt[j] = P_musc_t1 * np.exp(-(breath - t1) / tau) * (-1/tau)

        # need to interpolate for use in solve_ivp
        p_musc_func = interp1d(times, P_musc, kind='linear', fill_value='extrapolate')

        V0 = [0.0]
        t_span = (times[0], times[-1])
        t_eval = times

        solution = solve_ivp(self.dV_dt_function, t_span, V0, t_eval=t_eval, max_step = 0.003, method="RK23", rtol=1e-3,
                             atol=1e-6, args= (params, p_musc_func))

        V_time_array = solution.t
        volume_signal = solution.y
        dV_dt_values = self.dV_dt_function(V_time_array, volume_signal, params, p_musc_func)
        dV2_dt2_values_squared = ((1 / params["R_rs"]) * ((np.array(dP_musc_dt) - params["P_ao"]) - params["E_rs"] * dV_dt_values)) ** 2

        E1_n = (1 - np.array(P_musc) / Pmax) ** n
        E2_n = (1 - np.array(dP_musc_dt) / Pmax_dot) ** n

        inspire_index = np.argmin(np.abs((times % (t1 + t2)) - t1))

        integrand_inspire = P_musc[:inspire_index] * dV_dt_values[:inspire_index] / (E1_n[:inspire_index] * E2_n[:inspire_index]) + lambda1 * dV2_dt2_values_squared[:inspire_index]
        integrand_expire = dV2_dt2_values_squared[inspire_index:]

        # Integrate over time using Simpson’s rule
        integral_inspire = simpson(integrand_inspire, times[:inspire_index])
        integral_expire = simpson(integrand_expire, times[inspire_index:])

        WI = (1 / (t1 + t2)) * integral_inspire
        WE = (1 / (t1 + t2)) * integral_expire

        self.latest_volume = volume_signal[inspire_index]

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



