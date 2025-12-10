# import numpy as np
# from matplotlib import pyplot as plt
# # from matplotlib import pyplot as plt
# from scipy.integrate import simpson
# from numba import njit
#
# @njit
# def smooth_switch(times, t1, width):
#     """
#     Smooth approximation of Heaviside(t1 - t):
#     1 before t1, 0 after t1, with transition of scale 'width'.
#     """
#     n_times = len(times)
#     s = np.empty(n_times)
#     for i in range(n_times):
#         x = (times[i] - t1) / width
#         # avoid overflow
#         if x > 50.0:
#             s[i] = 0.0
#         elif x < -50.0:
#             s[i] = 1.0
#         else:
#             s[i] = 1.0 / (1.0 + np.exp(x))
#     return s
#
#
# @njit
# def compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance):
#     """
#     Compute constants for the respiratory model
#     """
#     # Precompute key values
#     a2 = (-P_ao - E_rs * VA * (t1 + t2) - E_rs * VD) / (t1 ** 2)
#     a1 = -2 * a2 * t1
#     Pt1 = a1 * t1 + a2 * (t1 ** 2)
#     Vt1 = VA * (t1 + t2) + VD
#     tau = max((t2 / (-np.log(tolerance * R_rs / Pt1))), 0.001)
#     B = E_rs / R_rs
#
#     return a1, a2, Pt1, Vt1, tau, B
#
#
# @njit
# def calculate_V_dV_dt(times, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao, width):
#     """
#     Updated method for calculating V and dV/dt values
#     """
#     # Precompute constants
#     t1, t2 = initial_guess
#     a1, a2, Pt1, Vt1, tau, B = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)
#
#     n_times = len(times)
#     V_insp = np.zeros(n_times)
#     V_exp = np.zeros(n_times)
#
#
#
#     # Breathing cycle patterns
#     mask_0_t1 = times <= t1
#     mask_t1_t2 = ~mask_0_t1
#
#     x = times[mask_0_t1]
#     z = times[mask_t1_t2]
#
#     # Compute constants for solution
#     c1 = (Vt1 - ((a1 / E_rs) * t1 + (a2 / E_rs) * (t1 ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t1)) / (np.exp(-B * t1) - 1)
#
#     d1 = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) - c1
#     c2 = (Vt1 - (Pt1 / R_rs) / (B - 1 / tau)) / np.exp(-B * t1)
#
#     # Calculate for 0 <= times <= t1
#     V_insp[mask_0_t1] = ((a1 / E_rs) * x - (a1 * R_rs / (E_rs ** 2)) +
#                     (a2 / E_rs) * (x ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * x +
#                     (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) +
#                     c1 * np.exp((-E_rs / R_rs) * x) + d1)
#
#     # Calculate for t1 <= times <= t1 + t2
#     constant = (Pt1 / (R_rs * (B - 1 / tau))) * np.exp(t1/tau)
#     V_exp[mask_t1_t2] = constant * np.exp(-z/tau) + np.exp(-B * z) * c2
#
#     # Smooth blend
#     s = smooth_switch(times, t1, width)
#     V = np.empty(n_times)
#
#     for i in range(n_times):
#         w_insp = s[i]
#         w_exp = 1.0 - w_insp
#         V[i] = w_insp * V_insp[i] + w_exp * V_exp[i]
#
#     return V
#
#
# @njit
# def calculate_single_V_dV_dt(t, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao):
#     """
#     Updated method for calculating V and dV/dt values
#     """
#     # Precompute constants
#     t1, t2 = initial_guess
#     a1, a2, Pt1, Vt1, tau, B = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)
#
#     # Compute constants for solution
#     c1 = (Vt1 - ((a1 / E_rs) * t1 + (a2 / E_rs) * (t1 ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t1)) / (
#                 np.exp(-B * t1) - 1)
#
#     d1 = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) - c1
#     c2 = (Vt1 - (Pt1 / R_rs) / (B - 1 / tau)) / np.exp(-B * t1)
#
#     if t <= t1:
#         V = ((a1 / E_rs) * t - (a1 * R_rs / (E_rs ** 2)) +
#                         (a2 / E_rs) * (t ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t +
#                         (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) +
#                         c1 * np.exp((-E_rs / R_rs) * t) + d1)
#         dV_dt = (1 / R_rs) * (a1 * t + a2 * (t ** 2) - E_rs * V)
#     else:
#         V = (Pt1 / (R_rs * (B - 1/tau))) * np.exp((-1/tau) * (t - t1)) + np.exp(-B * t) * c2
#         dV_dt = (1 / R_rs) * (Pt1 * np.exp(-(t - t1) / tau) - E_rs * V)
#
#     return V, dV_dt
#
# # @njit
# # def calculate_single_P_musc_dP_dt(t, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao):
# #     """
# #     Updated method for calculating P_musc and dP_musc/dt
# #     """
# #     t1, t2 = initial_guess
# #     a1, a2, Pt1, _, tau, _ = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)
# #
# #     if t <= t1: # Calculate P_musc for 0 <= times <= t1
# #         P_musc = a1 * t + a2 * (t ** 2)
# #         dP_musc_dt = a1 + 2 * a2 * t
# #     else:
# #         # Calculate P_musc for t1 <= times <= t1 + t2
# #         P_musc = Pt1 * np.exp(-(t - t1) / tau)
# #         dP_musc_dt = P_musc * (-1 / tau)
# #
# #     return P_musc, dP_musc_dt
#
# @njit
# def calculate_P_musc_dP_dt(times, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao, Pmax, width):
#     """
#     Updated method for calculating P_musc and dP_musc/dt
#     """
#     t1, t2 = initial_guess
#     a1, a2, Pt1, _, tau, _ = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)
#
#     n_times = len(times)
#     P_musc_insp = np.zeros(n_times)
#     dP_musc_dt_insp = np.zeros(n_times)
#     P_musc_exp = np.zeros(n_times)
#     dP_musc_dt_exp = np.zeros(n_times)
#     P_musc = np.zeros(n_times)
#     dP_musc_dt = np.zeros(n_times)
#
#     # Breathing cycle patterns
#     mask_0_t1 = times <= t1
#     mask_t1_t2 = ~mask_0_t1
#
#     # Calculate P_musc for 0 <= times <= t1
#     P_musc_insp[mask_0_t1] = a1 * times[mask_0_t1] + a2 * (times[mask_0_t1] ** 2)
#     dP_musc_dt_insp[mask_0_t1] = a1 + 2 * a2 * times[mask_0_t1]
#
#     # Calculate P_musc for t1 <= times <= t1 + t2
#     P_musc_exp[mask_t1_t2] = Pt1 * np.exp(-(times[mask_t1_t2] - t1) / tau)
#     P_musc_exp = np.minimum(P_musc_exp, Pmax)
#
#     dP_musc_dt_exp[mask_t1_t2] = P_musc_exp[mask_t1_t2] * (-1 / tau)
#
#     # Smooth switch weights
#     s = smooth_switch(times, t1, width)  # 1 ~ insp, 0 ~ exp
#
#     for i in range(n_times):
#         w_insp = s[i]
#         w_exp = 1.0 - w_insp
#         P_musc[i] = w_insp * P_musc_insp[i] + w_exp * P_musc_exp[i]
#         dP_musc_dt[i] = w_insp * dP_musc_dt_insp[i] + w_exp * dP_musc_dt_exp[i]
#
#
#     return P_musc, dP_musc_dt
#
#
# # @njit
# def objective(initial_guess, required_params, VAflow, VD, dt, tolerance):
#     """
#     Optimized Objective function
#     """
#     t1, t2 = initial_guess
#     n_steps = int(np.round((t1 + t2) / dt)) + 1
#     times = np.linspace(0, t1 + t2, n_steps)
#
#     lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao = required_params
#
#     width = 0.0001
#
#     P_musc, dP_musc_dt = calculate_P_musc_dP_dt(times, initial_guess, VAflow, VD, tolerance, E_rs, R_rs, P_ao, Pmax, width)
#     volume_signal = calculate_V_dV_dt(times, initial_guess, VAflow, VD, tolerance, E_rs, R_rs, P_ao, width)
#     dV_dt_values = (P_musc - P_ao - E_rs * volume_signal) / R_rs
#
#
#     inspire_index = int(np.round(t1 / dt))
#     dV2_dt2_values_squared = ((1 / R_rs) * ((dP_musc_dt - P_ao) -
#                                                       E_rs * dV_dt_values)) ** 2
#
#     E1_n = (1 - np.clip((P_musc / Pmax), 0, 0.999999)) ** n
#     E2_n = (1 - np.clip((np.abs(dP_musc_dt) / Pmax_dot), 0, 0.999999)) ** n
#
#     # Smooth inspiratory / expiratory weighting
#     s = smooth_switch(times, t1, width)
#     w_insp = s
#     w_exp = 1.0 - s
#
#     # print((-2 * a2 * t1), a2, tau, t1, t2)
#     plt.plot(volume_signal)
#     # plt.plot(check_V)
#     plt.show() # compare euler to analytical solution
#
#     # Compute inspiratory and expiratory integrals
#     with np.errstate(divide='ignore', invalid='ignore'):
#         integrand_inspire = w_insp * (P_musc * dV_dt_values) / (
#                 E1_n * E2_n) + lambda1 * dV2_dt2_values_squared
#
#     integrand_expire = w_exp * dV2_dt2_values_squared
#
#     integral_inspire = simpson(integrand_inspire)
#     integral_expire = simpson(integrand_expire)
#
#     WI = (1 / (t1 + t2)) * integral_inspire
#     WE = (1 / (t1 + t2)) * integral_expire
#
#     # Return cost function value
#     return WI + lambda2 * WE
#
#
#
#
#
#
#
#
#
# # # just smooth objective
# # import numpy as np
# # from matplotlib import pyplot as plt
# # from scipy.integrate import simpson
# # from numba import njit
# #
# #
# # @njit
# # def smooth_switch(times, t1, width):
# #     """
# #     Smooth approximation of Heaviside(t1 - t):
# #     1 before t1, 0 after t1, with transition of scale 'width'.
# #     """
# #     n_times = len(times)
# #     s = np.empty(n_times)
# #     for i in range(n_times):
# #         x = (times[i] - t1) / width
# #         # avoid overflow
# #         if x > 50.0:
# #             s[i] = 0.0
# #         elif x < -50.0:
# #             s[i] = 1.0
# #         else:
# #             s[i] = 1.0 / (1.0 + np.exp(x))
# #     return s
# #
# #
# # @njit
# # def compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance):
# #     """
# #     Compute constants for the respiratory model
# #     """
# #     # Precompute key values
# #     a2 = (-P_ao - E_rs * VA * (t1 + t2) - E_rs * VD) / (t1 ** 2)
# #     a1 = -2 * a2 * t1
# #     Pt1 = a1 * t1 + a2 * (t1 ** 2)
# #     Vt1 = VA * (t1 + t2) + VD
# #     tau = max((t2 / (-np.log(tolerance * R_rs / Pt1))), 0.001)
# #     B = E_rs / R_rs
# #
# #     return a1, a2, Pt1, Vt1, tau, B
# #
# #
# # @njit
# # def calculate_V_dV_dt(times, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao):
# #     """
# #     Updated method for calculating V and dV/dt values
# #     """
# #     # Precompute constants
# #     t1, t2 = initial_guess
# #     a1, a2, Pt1, Vt1, tau, B = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)
# #
# #     V = np.zeros(len(times))
# #
# #     # Breathing cycle patterns
# #     mask_0_t1 = times <= t1
# #     mask_t1_t2 = ~mask_0_t1
# #
# #     x = times[mask_0_t1]
# #     z = times[mask_t1_t2]
# #
# #     # Compute constants for solution
# #     c1 = (Vt1 - ((a1 / E_rs) * t1 + (a2 / E_rs) * (t1 ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t1)) / (
# #                 np.exp(-B * t1) - 1)
# #
# #     d1 = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) - c1
# #     c2 = (Vt1 - (Pt1 / R_rs) / (B - 1 / tau)) / np.exp(-B * t1)
# #
# #     # Calculate for 0 <= times <= t1
# #     V[mask_0_t1] = ((a1 / E_rs) * x - (a1 * R_rs / (E_rs ** 2)) +
# #                     (a2 / E_rs) * (x ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * x +
# #                     (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) +
# #                     c1 * np.exp((-E_rs / R_rs) * x) + d1)
# #
# #     # Calculate for t1 <= times <= t1 + t2
# #     constant = (Pt1 / (R_rs * (B - 1 / tau))) * np.exp(t1 / tau)
# #     V[mask_t1_t2] = constant * np.exp(-z / tau) + np.exp(-B * z) * c2
# #
# #     return V
# #
# #
# # @njit
# # def calculate_single_V_dV_dt(t, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao):
# #     """
# #     Updated method for calculating V and dV/dt values
# #     """
# #     # Precompute constants
# #     t1, t2 = initial_guess
# #     a1, a2, Pt1, Vt1, tau, B = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)
# #
# #     # Compute constants for solution
# #     c1 = (Vt1 - ((a1 / E_rs) * t1 + (a2 / E_rs) * (t1 ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t1)) / (
# #             np.exp(-B * t1) - 1)
# #
# #     d1 = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) - c1
# #     c2 = (Vt1 - (Pt1 / R_rs) / (B - 1 / tau)) / np.exp(-B * t1)
# #
# #     if t <= t1:
# #         V = ((a1 / E_rs) * t - (a1 * R_rs / (E_rs ** 2)) +
# #              (a2 / E_rs) * (t ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t +
# #              (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) +
# #              c1 * np.exp((-E_rs / R_rs) * t) + d1)
# #         dV_dt = (1 / R_rs) * (a1 * t + a2 * (t ** 2) - E_rs * V)
# #     else:
# #         V = (Pt1 / (R_rs * (B - 1 / tau))) * np.exp((-1 / tau) * (t - t1)) + np.exp(-B * t) * c2
# #         dV_dt = (1 / R_rs) * (Pt1 * np.exp(-(t - t1) / tau) - E_rs * V)
# #
# #     return V, dV_dt
# #
# #
# # # @njit
# # # def calculate_single_P_musc_dP_dt(t, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao):
# # #     """
# # #     Updated method for calculating P_musc and dP_musc/dt
# # #     """
# # #     t1, t2 = initial_guess
# # #     a1, a2, Pt1, _, tau, _ = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)
# # #
# # #     if t <= t1: # Calculate P_musc for 0 <= times <= t1
# # #         P_musc = a1 * t + a2 * (t ** 2)
# # #         dP_musc_dt = a1 + 2 * a2 * t
# # #     else:
# # #         # Calculate P_musc for t1 <= times <= t1 + t2
# # #         P_musc = Pt1 * np.exp(-(t - t1) / tau)
# # #         dP_musc_dt = P_musc * (-1 / tau)
# # #
# # #     return P_musc, dP_musc_dt
# #
# # @njit
# # def calculate_P_musc_dP_dt(times, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao, Pmax):
# #     """
# #     Updated method for calculating P_musc and dP_musc/dt
# #     """
# #     t1, t2 = initial_guess
# #     a1, a2, Pt1, _, tau, _ = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)
# #
# #     P_musc = np.zeros(len(times))
# #     dP_musc_dt = np.zeros(len(times))
# #
# #     # Breathing cycle patterns
# #     mask_0_t1 = times <= t1
# #     mask_t1_t2 = ~mask_0_t1
# #
# #     # Calculate P_musc for 0 <= times <= t1
# #     P_musc[mask_0_t1] = a1 * times[mask_0_t1] + a2 * (times[mask_0_t1] ** 2)
# #     dP_musc_dt[mask_0_t1] = a1 + 2 * a2 * times[mask_0_t1]
# #
# #     # Calculate P_musc for t1 <= times <= t1 + t2
# #     P_musc[mask_t1_t2] = Pt1 * np.exp(-(times[mask_t1_t2] - t1) / tau)
# #     P_musc = np.minimum(P_musc, Pmax)
# #
# #     dP_musc_dt[mask_t1_t2] = P_musc[mask_t1_t2] * (-1 / tau)
# #
# #     return P_musc, dP_musc_dt
# #
# #
# # # @njit
# # def objective(initial_guess, required_params, VAflow, VD, dt, tolerance):
# #     """
# #     Optimized Objective function
# #     """
# #     t1, t2 = initial_guess
# #     n_steps = int(np.round((t1 + t2) / dt)) + 1
# #     times = np.linspace(0, t1 + t2, n_steps)
# #
# #     lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao = required_params
# #
# #     P_musc, dP_musc_dt = calculate_P_musc_dP_dt(times, initial_guess, VAflow, VD, tolerance, E_rs, R_rs, P_ao, Pmax)
# #     volume_signal = calculate_V_dV_dt(times, initial_guess, VAflow, VD, tolerance, E_rs, R_rs, P_ao)
# #     dV_dt_values = (P_musc - P_ao - E_rs * volume_signal) / R_rs
# #
# #     inspire_index = int(np.round(t1 / dt))
# #     dV2_dt2_values_squared = ((1 / R_rs) * ((dP_musc_dt - P_ao) -
# #                                             E_rs * dV_dt_values)) ** 2
# #
# #     # E1_n = (1 - np.clip((P_musc / Pmax), 0, 0.999999)) ** n
# #     # E2_n = (1 - np.clip((np.abs(dP_musc_dt) / Pmax_dot), 0, 0.999999)) ** n
# #
# #     E1_n = (1 - (P_musc / Pmax)) ** n
# #     E2_n = (1 - (np.abs(dP_musc_dt) / Pmax_dot)) ** n
# #
# #     # Smooth inspiratory / expiratory weighting
# #     width = 0.02
# #     s = smooth_switch(times, t1, width)
# #     w_insp = s
# #     w_exp = 1.0 - s
# #
# #     # # print((-2 * a2 * t1), a2, tau, t1, t2)
# #     # plt.plot(dV2_dt2_values_squared)
# #     # # plt.plot(check_V)
# #     # plt.show() # compare euler to analytical solution
# #
# #     # Compute inspiratory and expiratory integrals
# #     with np.errstate(divide='ignore', invalid='ignore'):
# #         integrand_inspire = w_insp * ((P_musc * dV_dt_values) / (
# #                 E1_n * E2_n) + lambda1 * dV2_dt2_values_squared)
# #
# #     integrand_expire = w_exp * dV2_dt2_values_squared
# #
# #     integral_inspire = simpson(integrand_inspire)
# #     integral_expire = simpson(integrand_expire)
# #
# #     WI = (1 / (t1 + t2)) * integral_inspire
# #     WE = (1 / (t1 + t2)) * integral_expire
# #
# #     # Return cost function value
# #     return WI + lambda2 * WE
