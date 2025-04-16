import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import interp1d


class BreathOptimiser:
    def __init__(self, params, VAflow, VD, dt):
        self.params = params
        self.VA = VAflow
        self.VD = VD
        self.dt = dt

    # def constraint_function(self, initial_Nd_guess, VD, VA):
    #     [a2, tau, t1] = initial_Nd_guess
    #     params = self.params
    #
    #     a1 = -2 * a2 * t1
    #     t2 = (-a2 * (t1) ** 2 - params["P_ao"] - params["E_rs"] * self.VA * t1 - params["E_rs"] * self.VD) / (params["E_rs"] * self.VA)
    #
    #     constraint = (a1 * t1 + a2 * (t1 **2) - params["P_ao"]) - params["E_rs"] * (VA * (t1 + t2) + VD)
    #     return constraint

    def tau_constraint(self, initial_Nd_guess):
        [a2, tau, t1] = initial_Nd_guess
        params = self.params

        a1 = -2 * a2 * t1
        t2 = (-a2 * (t1) ** 2 - params["P_ao"] - params["E_rs"] * self.VA * t1
              - params["E_rs"] * self.VD) / (params["E_rs"] * self.VA)

        Pt1 = a1 * t1 + a2 * (t1 ** 2)
        constraint = Pt1 * np.exp(-t2 / tau)
        return constraint

    def simulate_euler(self, times, P_musc):
        params = self.params
        V = np.zeros(len(times))
        dV_dt = np.zeros(len(times))
        dt = np.diff(times)

        for i in range(1, len(times)):
            dV_dt[i-1] = (1 / params["R_rs"]) * ((P_musc[i-1] - params["P_ao"]) - params["E_rs"] * V[i - 1])
            V[i] = V[i - 1] + dV_dt[i-1] * dt[i - 1]

        dV_dt[i] = (1 / params["R_rs"]) * ((P_musc[i] - params["P_ao"]) - params["E_rs"] * V[i]) # no need to calculate as dV_dt should be 0 at the end
        # plt.plot(times, dt, marker='o')
        # plt.show()

        return V, dV_dt


    def simulate_euler_interpolated(self, times, P_musc):
        params = self.params
        V = np.zeros(len(times))
        dV_dt = np.zeros(len(times))
        dt = np.diff(times)

        for i in range(1, len(times)):
            t_prev = times[i - 1]
            dV_dt[i-1] = (1 / params["R_rs"]) * ((P_musc(t_prev) - params["P_ao"]) - params["E_rs"] * V[i - 1])
            V[i] = V[i - 1] + dV_dt[i-1] * dt[i - 1]

        dV_dt[i] = (1 / params["R_rs"]) * ((P_musc(times[i]) - params["P_ao"]) - params["E_rs"] * V[i])

        # plt.plot(times, dt, marker='o')
        # plt.show()

        return V, dV_dt


    def simulate_rk4(self, times, P_musc):
        params = self.params
        V = np.zeros(len(times))
        dV_dt = np.zeros(len(times))
        dt = np.diff(times)

        def dV(V_local, t_local):
            return (1 / params["R_rs"]) * ((P_musc(t_local) - params["P_ao"]) - params["E_rs"] * V_local)

        for i in range(1, len(times)):
            dt_prev = dt[i-1] # sensitive to whether it is dt or 0.001
            t_prev = times[i - 1]
            V_prev = V[i - 1]

            k1 = dV(V_prev, t_prev)
            k2 = dV(V_prev + 0.5 * dt_prev * k1, t_prev + 0.5 * dt_prev)
            k3 = dV(V_prev + 0.5 * dt_prev * k2, t_prev + 0.5 * dt_prev)
            k4 = dV(V_prev + dt_prev * k3, t_prev + dt_prev)

            V[i] = V_prev + (dt_prev / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            dV_dt[i] = (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return V, dV_dt

    def simulate_rk_scipy(self, times, P_musc):
        params = self.params
        P_interp = interp1d(times, P_musc, kind='linear', fill_value='extrapolate')

        def dVdt(t, V):
            return (1 / params["R_rs"]) * ((P_interp(t) - params["P_ao"]) - params["E_rs"] * V)

        # Use solve_ivp with RK45 (adaptive Runge-Kutta)
        sol = solve_ivp(
            dVdt,
            t_span=(times[0], times[-1]),
            y0=[0],  # assuming V(0) = 0
            t_eval=times,  # force solver to return values at these times
            method='RK45'
        )

        V = sol.y[0]
        dV_dt = np.array([dVdt(t, v) for t, v in zip(times, V)])

        return V, dV_dt


    def calculate_P_musc_dP_dt(self, times, initial_Nd_guess, t2):
        [a2, tau, t1] = initial_Nd_guess
        a1 = -2 * a2 * t1

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

        return P_musc, dP_musc_dt


    def objective(self, initial_Nd_guess):
        """
         Function to obtain a0, a1, a2, tau, t1, t2 Edit: removed a0 and a1
        """
        [a2, tau, t1] = initial_Nd_guess
        params = self.params

        t2 = ((-a2 * (t1) ** 2 - params["P_ao"] - params["E_rs"] * self.VA * t1 - params["E_rs"] * self.VD) /
              (params["E_rs"] * self.VA))

        if t2 < 0:
            return np.inf

        # if t2 < 0:
        #     print(f"constraint: {(-a2 * (t1 ** 2) - params["P_ao"] - params["E_rs"] * self.VA * t1 - params["E_rs"] * self.VD)/ (params["E_rs"] * self.VA)}")
        #     print(self.VA)
        #     print(self.VD)
        #     print(a2)
        #     print(t1)
        #     raise ValueError(f"Error: t2 cannot be less than 0. Received t2 = {t2}.")

        n_steps = int(np.round((t1 + t2) / self.dt)) + 1
        times = np.linspace(0, (t1 + t2), n_steps)

        # Breathing Pattern Optimiser
        lambda1 = params["lambda1"]
        lambda2 = params["lambda2"]
        n = params["n"]
        Pmax = params["Pmax"]
        Pmax_dot = params["Pmax_dot"]

        P_musc, dP_musc_dt = self.calculate_P_musc_dP_dt(times, initial_Nd_guess, t2)

        # # Interpolate P_musc for intermediate time evaluations
        # P_interp = interp1d(times, P_musc, kind="linear", fill_value="extrapolate")
        # volume_signal, dV_dt_values = self.simulate_rk4(times, P_interp)
        # volume_signal, dV_dt_values  = self.simulate_euler_interpolated(times, P_interp)

        volume_signal, dV_dt_values  = self.simulate_euler(times, P_musc)


        # print(f"guess: {initial_Nd_guess}")

        inspire_index = int(round(t1 / self.dt))
        dV2_dt2_values_squared = ((1 / params["R_rs"]) * ((dP_musc_dt - params["P_ao"]) - params["E_rs"] * dV_dt_values)) ** 2

        E1_n = (1 - P_musc / Pmax) ** n
        E2_n = (1 - dP_musc_dt / Pmax_dot) ** n

        integrand_inspire = (P_musc[:inspire_index] * dV_dt_values[:inspire_index]) / (E1_n[:inspire_index] * E2_n[:inspire_index]) + lambda1 * dV2_dt2_values_squared[:inspire_index]
        integrand_expire = dV2_dt2_values_squared[inspire_index:]

        # print(f"integrand_inspire.shape: {integrand_inspire.shape}")
        # print(f"times[:inspire_index].shape: {times[:inspire_index].shape}")
        # print(f"integrand_expire.shape: {integrand_expire.shape}")
        # print(f"times[inspire_index:].shape: {times[inspire_index:].shape}")

        # Integrate over time using Simpson’s rule
        integral_inspire = simpson(integrand_inspire, x=times[:inspire_index])
        integral_expire = simpson(integrand_expire, x=times[inspire_index:])

        WI = (1 / (t1 + t2)) * integral_inspire
        WE = (1 / (t1 + t2)) * integral_expire

        J = WI + lambda2 * WE

        return J