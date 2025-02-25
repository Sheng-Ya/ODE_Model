# def elastance(t_i, tr, td):
#     E_p = np.where(t_i <= tr,
#                    0.5 * (1.0 - np.cos(np.pi * t_i / tr)),
#                    np.where(t_i <= td,
#                             0.5 * (1.0 + np.cos(np.pi * (t_i - tr) / (td - tr))),
#                             0))
#
#     DE_p = np.where(t_i <= tr,
#                     0.5 * np.pi / tr * np.sin(np.pi * t_i / tr),
#                     np.where(t_i <= td,
#                              -0.5 * np.pi / (td - tr) * np.sin(np.pi * (t_i - tr)
#
#     tr = 0.3 * τ
#     td = 0.45 * τ


# # # State variables calculated in the PULMONARY circulation
# #     VT_pa = state['VT_pa']
# #     VT_pp = state['VT_pp']
# #     VT_pv = state['VT_pv']
# #     Q_pa = state['Q_pa']
# #
# #     # State variables calculated in the HEART circulation
# #     VT_la = state['VT_la']
# #     VT_lv = state['VT_lv']
# #     VT_ra = state['VT_ra']
# #     VT_rv = state['VT_rv']
# #
# #     # State variables calculated in the SYSTEMIC circulation
# #     VT_sv = state["VT_sv"]
# #     VT_bv = state["VT_bv"]
# #     VT_hv = state["VT_hv"]
# #     VT_rmv = state["VT_rmv"]
# #     VT_amv = state["VT_amv"]
# #     VT_ev = state["VT_ev"]
# #     P_sp = state["P_sp"]
# #     V_sa = state["V_sa"]
# #     P_sa = state["P_sa"]
# #     Q_sa = state["Q_sa"]
# #     VT_vc = state["VT_vc"]
#
#
#
#
#
#
#
# # Respiratory mechanics
# def respiratory_system(t, state, params):
#     """
#     Respiratory mechanics dynamics from supplementary PDF.
#     State variables:
#     V_lung - Lung volume
#     P_airway - Airway pressure
#     """
#
#
#     # Time-varying elastance for the left ventricle
#     E_lv = E_max_lv * (0.5 * (1 - np.cos(2 * np.pi * (t % 1.0))))  # Simplified cycle
#
#     # Pressure-volume relationship for the left ventricle
#     P_lv = E_lv * (V_lv - V0_lv)
#
#     # Flow dynamics
#     Q_lv = (P_lv - P_sa) / R_lv  # Flow out of left ventricle
#     Q_sa = P_sa / R_sa  # Flow through systemic arteries
#
#     # Differential equations
#     dP_sa_dt = (Q_lv - Q_sa) / C_sa
#     dV_lv_dt = -Q_lv
#
#
#
#
#
#     V_lung, P_airway = state
#     R_aw = params["R_aw"]
#     C_aw = params["C_aw"]
#
#     # Flow dynamics
#     Q_air = (P_airway - 0) / R_aw  # Assuming atmospheric pressure = 0
#
#     # Differential equations
#     dV_lung_dt = Q_air
#     dP_airway_dt = (1 / C_aw) * (V_lung - P_airway)
#
#     return [dV_lung_dt, dP_airway_dt]
#
# # Gas exchange
# def gas_exchange_system(t, state, params):
#     """
#     Gas exchange dynamics from supplementary PDF.
#     State variables:
#     PaO2 - Arterial oxygen pressure
#     PaCO2 - Arterial carbon dioxide pressure
#     """
#     PaO2, PaCO2 = state
#     O2_consumption = params["O2_consumption"]
#     CO2_production = params["CO2_production"]
#
#     # Gas exchange equations
#     dPaO2_dt = -O2_consumption / 100  # Simplified oxygen consumption
#     dPaCO2_dt = CO2_production / 100  # Simplified CO2 production
#
#     return [dPaO2_dt, dPaCO2_dt]
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # Combined system dynamics
# def combined_system(t, state, params):
#     """
#     Combines cardiovascular, respiratory, and gas exchange systems.
#     State variables:
#     P_sa, V_lv, V_lung, P_airway, PaO2, PaCO2
#     """
#     cardio_state = state[:2]
#     resp_state = state[2:4]
#     gas_state = state[4:6]
#
#     # Feedback control
#     HR_mod, Resp_mod = feedback_controller(cardio_state[0], gas_state[0], gas_state[1], params)
#
#     # Cardiovascular dynamics
#     d_cardio = cardiovascular_system(t, cardio_state, params)
#
#     # Respiratory dynamics (with feedback modulation)
#     d_resp = respiratory_system(t, resp_state, params)
#
#     # Gas exchange dynamics
#     d_gas = gas_exchange_system(t, gas_state, params)
#
#     return d_cardio + d_resp + d_gas
#
# # Simulation
# def simulate():
#     # Initial conditions
#     init_state = [
#         80.0, 120.0,  # Cardiovascular: P_sa, V_lv
#         0.5, 0.0,     # Respiratory: V_lung, P_airway
#         100.0, 40.0   # Gas exchange: PaO2, PaCO2
#     ]
#
#     # Time span
#     t_span = (0, 10)  # Simulate for 10 seconds
#     t_eval = np.linspace(t_span[0], t_span[1], 1000)
#
#     # Solve ODE
#     solution = solve_ivp(combined_system, t_span, init_state, t_eval=t_eval, args=(PARAMS,))
#
#     return solution
#
# # Run simulation and plot results
# if __name__ == "__main__":
#     solution = simulate()
#
#     # Extract results
#     t = solution.t
#     P_sa, V_lv = solution.y[0], solution.y[1]
#     V_lung, P_airway = solution.y[2], solution.y[3]
#     PaO2, PaCO2 = solution.y[4], solution.y[5]
#
#
#
#
#
#
#
#
#
#
#
#
#
# import numpy as np
# from scipy.integrate import solve_ivp
#
# from Gas_Exchange import gas_exchange
# from Initial_Conditions import Initial_Conditions
# from Parameters import Parameters
#
# # First iteration
# # get the first derivative and outputs from all the separated systems
# def combined_system(t, Initial_Conditions_numpy, Parameters, Initial_Conditions_dict):
#     """
#
#     """
#     time_history = []
#     beta = []
#     U = []
#
#     # Cardiovascular dynamics
#     d_gas = gas_exchange(t, Initial_Conditions_numpy, Parameters, time_history, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict)
#
#     return
#
#
# def simulate():
#
#     # Time span
#     t_span = (0, 0.001)  # Simulate for x seconds
#     t_eval = np.linspace(t_span[0], t_span[1], 2)
#
#     required_gas_keys = ["Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2", "Pd_4_O2", "Pd_4_CO2",
#                          "Pd_5_O2", "Pd_5_CO2", "Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2",
#                          "PvbCO2", "PCSFCO2", "MRTO2", "MRTCO2", "CvO2", "CvCO2", "MRV"]
#     IC_gas = np.array([Initial_Conditions[key] for key in required_gas_keys], dtype=float)
#
#     # Solve ODE
#     ODE_solution = solve_ivp(combined_system, t_span, IC_gas, method="RK23", t_eval=t_eval, max_step=0.01, rtol=1e-3, atol=1e-6, args=(Parameters, Initial_Conditions))
#     # print("Time Points:", ODE_solution.t)
#     # print("State Variables at Time Points:", ODE_solution.y)
#
#     return ODE_solution
#
#
#
# if __name__ == "__main__":
#     solution = simulate()
#
#
#
#
#
#
#
#
#
#
#
# import numpy as np
#
# global_results = {}
#
#
#
#
#
# def breath_optimiser(initial_Nd_guess, t, time_history, params, exp_inputs, resp_mech_inputs):
#     """
#      Function to obtain a0, a1, a2, tau, t1, t2
#      Other inputs: step_size, previous dV_dt, current dV_dt, P_musc, previous WI, previous WE
#
#     """
#     [a0, a1, a2, tau, t1, t2] = initial_Nd_guess
#
#     # Breathing Pattern Optimiser
#     lambda1 = params["lambda1"]
#     lambda2 = params["lambda2"]
#     n = params["n"]
#     Pmax = params["Pmax"]
#     Pmax_dot = params["Pmax_dot"]
#
#     # other inputs
#     dV_dt = resp_mech_inputs["dV_dt"]
#     previous_dV_dt = exp_inputs["previous_dV_dt"][-1]
#     P_musc = resp_mech_inputs["P_musc"]
#     previous_WI = exp_inputs["previous_WI"][-1]
#     previous_WE = exp_inputs["previous_WE"][-1]
#
#     if previous_WI and previous_WI[-1] != []:
#         previous_WI = previous_WI[-1]
#     else:
#         previous_WI = 0
#
#     if previous_WE and previous_WE[-1] != []:
#         previous_WE = previous_WE[-1]
#     else:
#         previous_WE = 0
#
#
#     if time_history and time_history[-1] != []:
#         step_size = t - time_history[-1]
#     else:
#         step_size = t
#
#     breath = t % (t1 + t2)
#
#     if 0 <= breath <= t1:
#         dP_musc_dt = a1 + 2 * a2 * t
#     elif t1 < breath <= (t1 + t2):
#         P_musc_t1 = a0 + a1 * t1 + a2 * (t1 ** 2)
#         dP_musc_dt = P_musc_t1 * np.exp(-(t - t1) / tau) * (-1 / tau)
#
#     d2V_dt2_squared = ((previous_dV_dt - dV_dt) / step_size) ** 2
#
#     E1 = (1 - P_musc / Pmax) ** n
#     E2 = (1 - dP_musc_dt / Pmax_dot) ** n
#
#     WI = previous_WI
#     WE = previous_WE
#
#     if 0 <= breath <= t1:
#         dWI_dt = (1/(t1+t2)) * (P_musc * dV_dt / (E1 * E2)) + lambda1 * d2V_dt2_squared
#         WI = previous_WI + dWI_dt * step_size  # Integrate using Euler's method
#     else:
#         dWE_dt = (1/(t1+t2)) * d2V_dt2_squared
#         WE = previous_WE + dWE_dt * step_size
#
#
#     previous_WI = WI
#     previous_WE = WE
#
#     J = WI + lambda2 * WE
#
#     # Store WI and WE globally
#     global_results['WI'] = WI
#     global_results['WE'] = WE
#     global_results['previous_WI'] = previous_WI
#     global_results['previous_WE'] = previous_WE
#
#     time_history.append(t)
#
#     return J
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import numpy as np
#
# def respiratory_mechanics(t, state, params, exp_inputs, updates):
#     """
#         Pulmonary Mechanics state variables: V
#         Upper Airway state variables: alpha
#
#     """
#
#     (V, alpha) = state
#
#     ## Pulmonary Mechanics
#     E_CW = params["E_CW"]
#     E_L = params["E_L"]
#     E_rs = params["E_rs"]
#     k_aw1 = params["k_aw1"]
#     k_aw2 = params["k_aw2"]
#     P_ao = params["P_ao"]
#     R_rs = params["R_rs"]
#
#     ## Upper Airways
#     A0_ua = params["A0_ua"]
#     b_ua = params["b_ua"]
#     C_ua = params["C_ua"]
#     K_ua = params["K_ua"]
#     Pcrit_min = params["Pcrit_min"]
#     R_AW = params["R_AW"]
#     R_CW = params["R_CW"]
#     R_L = params["R_L"]
#     R_trachea = params["R_trachea"]
#
#     previous_dV_dt = exp_inputs["previous_dV_dt"][-1]
#     a0, a1, a2, tau, t1, t2 = exp_inputs["Nd"][-6:]
#
#     E_rs = E_CW + E_L
#
#     breath = t % (t1 + t2)
#
#     if 0 <= breath <= t1:
#         P_musc = a0 + a1 * breath + a2 * (breath ** 2)
#     elif t1 < breath <= (t1 + t2):
#         P_musc_t1 = a0 + a1 * t1 + a2 * (t1 ** 2)
#         P_musc = P_musc_t1 * np.exp(-(breath - t1) / tau)
#
#
#     # initial value for G_AW
#     G_AW = exp_inputs["G_AW_guess"][-1]
#     Vflow_ua = exp_inputs["Vflow_ua"][-1]
#     P_ua = exp_inputs["P_ua"][-1]
#     max_iterations = 20
#
#     # Iterative calculation for G_AW
#     for _ in range(max_iterations):
#
#         # Calculate dV/dt using the current G_AW, minute ventilation = dV/dt
#         dV_dt = (G_AW / R_rs) * ((P_musc - P_ao) - E_rs * V)
#
#         if dV_dt < 0:
#             P_CW = E_CW * V - 1
#             P_a_dash = P_ao
#         else:
#             P_CW = E_CW * V - 1 + R_CW * dV_dt
#             P_a_dash = P_ao - k_aw1 * dV_dt - k_aw2 * (np.abs(dV_dt)) ** 2
#
#         if P_a_dash < 0:
#             P_a = 0
#         else:
#             P_a = P_a_dash
#
#         P_pl = P_CW + P_a - P_musc
#
#         tolerance = 1e-6
#
#         for _ in range(max_iterations):
#             # Airway pressure and flow
#             Vflow_LA = Vflow_ua + dV_dt
#             new_P_ua = P_pl + Vflow_LA * R_rs
#             new_Vflow_ua = -(1 / R_trachea) * (new_P_ua + (1 / C_ua) * alpha)
#             if abs(new_Vflow_ua - Vflow_ua) < tolerance and abs(new_P_ua - P_ua) < tolerance:
#                 Vflow_ua = new_Vflow_ua
#                 P_ua = new_P_ua
#                 break
#             Vflow_ua = new_Vflow_ua
#             P_ua = new_P_ua
#
#         # Set based on fixed parameters
#         Pcrit = Pcrit_min
#
#         # Update G_AW
#         if P_ua <= Pcrit:
#             new_G_AW = 0
#         elif (Pcrit < P_ua <= 0) and (1 - (P_ua / Pcrit)) >= 0:
#             new_G_AW = A0_ua * (1 - (P_ua / Pcrit)) * K_ua
#         elif P_ua > 0:
#             new_G_AW = A0_ua * K_ua
#
#         # Convergence check
#         if abs(new_G_AW - G_AW) < tolerance:
#             G_AW = new_G_AW
#             break
#         G_AW = new_G_AW
#
#
#     # known: P_pl, R_rs, C_ua, R_trachea
#     # solving for V_flow_LA, Vflow_ua, P_ua
#
#     # Vflow_LA = (P_ua - P_pl) / R_rs
#     # Vflow_ua = Vflow_LA - dV_dt
#     # P_ua = (-R_trachea) * Vflow_ua - alpha / C_ua
#     # Vflow_ua = (1/(1+R_trachea/R_rs)) * ((- (1/(R_rs * C_ua)) * alpha) - P_pl/R_rs - dV_dt)
#
#     d_alpha_dt = Vflow_ua
#     # R_rs = R_AW + R_L + R_CW
#
#     # t_eval = updates["t_eval6"][0]
#     # tolerance = 1e-3
#     # if np.abs(t - t_eval) < tolerance:
#     exp_inputs["G_AW_guess"].append(G_AW)
#     exp_inputs["Vflow_ua"].append(Vflow_ua)
#     exp_inputs["P_ua"].append(P_ua)
#
#     updates["G_AW"].append(G_AW)
#     updates["Vflow_ua"].append(Vflow_ua)
#     updates["P_ua"].append(P_ua)
#     updates["P_musc"].append(P_musc)
#     updates["dV_dt"].append(dV_dt)
#     updates["V"].append(V)
#     updates["previous_dV_dt"].append(dV_dt)
#     updates["P_pl"].append(P_pl)
#
#     # updates["t_eval6"] = updates["t_eval6"][1:]
#
#     return [dV_dt, d_alpha_dt]
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import numpy as np
#
# def respiratory_mechanics(t, state, params, exp_inputs, updates):
#     """
#         Pulmonary Mechanics state variables: V
#         Upper Airway state variables: alpha
#
#     """
#
#     (V, alpha) = state
#
#     ## Pulmonary Mechanics
#     E_CW = params["E_CW"]
#     E_L = params["E_L"]
#     E_rs = params["E_rs"]
#     k_aw1 = params["k_aw1"]
#     k_aw2 = params["k_aw2"]
#     P_ao = params["P_ao"]
#     R_rs = params["R_rs"]
#
#     ## Upper Airways
#     A0_ua = params["A0_ua"]
#     b_ua = params["b_ua"]
#     C_ua = params["C_ua"]
#     K_ua = params["K_ua"]
#     Pcrit_min = params["Pcrit_min"]
#     R_AW = params["R_AW"]
#     R_CW = params["R_CW"]
#     R_L = params["R_L"]
#     R_trachea = params["R_trachea"]
#
#     a0, a1, a2, tau, t1, t2 = exp_inputs["Nd"][-6:]
#
#     E_rs = E_CW + E_L
#
#     breath = t % (t1 + t2)
#
#     if 0 <= breath <= t1:
#         P_musc = a0 + a1 * breath + a2 * (breath ** 2)
#     elif t1 < breath <= (t1 + t2):
#         P_musc_t1 = a0 + a1 * t1 + a2 * (t1 ** 2)
#         P_musc = P_musc_t1 * np.exp(-(breath - t1) / tau)
#
#
#     # initial value for G_AW
#     G_AW = exp_inputs["G_AW"][-1]
#     Vflow_ua = exp_inputs["Vflow_ua"][-1]
#
#     # got rid of Iterative calculation for G_AW
#
#
#     # Calculate dV/dt using the current G_AW, minute ventilation = dV/dt
#     dV_dt = (G_AW / R_rs) * ((P_musc - P_ao) - E_rs * V)
#
#     if dV_dt < 0:
#         P_CW = E_CW * V - 1
#         P_a_dash = P_ao
#     else:
#         P_CW = E_CW * V - 1 + R_CW * dV_dt
#         P_a_dash = P_ao - k_aw1 * dV_dt - k_aw2 * (np.abs(dV_dt)) ** 2
#
#     if P_a_dash < 0:
#         P_a = 0
#     else:
#         P_a = P_a_dash
#
#     P_pl = P_CW + P_a - P_musc
#
#     tolerance = 1e-6
#
#     # Airway pressure and flow
#     Vflow_LA = Vflow_ua + dV_dt
#     P_ua = P_pl + Vflow_LA * R_rs
#
#     a = -(1 / R_trachea)
#     b = (P_ua + (1 / C_ua) * alpha)
#     Vflow_ua = -(1 / R_trachea) * (P_ua + ((1 / C_ua) * alpha))
#     c = Vflow_ua
#
#     # Set based on fixed parameters
#     Pcrit = Pcrit_min
#
#     # Update G_AW
#     if P_ua <= Pcrit:
#         G_AW = 0
#     elif (Pcrit < P_ua <= 0) and (1 - (P_ua / Pcrit)) >= 0:
#         G_AW = A0_ua * (1 - (P_ua / Pcrit)) * K_ua
#     elif P_ua > 0:
#         G_AW = A0_ua * K_ua
#
#     d_alpha_dt = Vflow_ua
#
#
#     # known: P_pl, R_rs, C_ua, R_trachea
#     # solving for V_flow_LA, Vflow_ua, P_ua
#
#     # Vflow_LA = (P_ua - P_pl) / R_rs
#     # Vflow_ua = Vflow_LA - dV_dt
#     # P_ua = (-R_trachea) * Vflow_ua - alpha / C_ua
#     # Vflow_ua = (1/(1+R_trachea/R_rs)) * ((- (1/(R_rs * C_ua)) * alpha) - P_pl/R_rs - dV_dt)
#
#     # R_rs = R_AW + R_L + R_CW
#
#     # t_eval = updates["t_eval6"][0]
#     # tolerance = 1e-3
#     # if np.abs(t - t_eval) < tolerance:
#
#     updates["G_AW"].append(G_AW)
#     updates["Vflow_ua"].append(Vflow_ua)
#     updates["P_ua"].append(P_ua)
#     updates["P_musc"].append(P_musc)
#     updates["dV_dt"].append(dV_dt)
#     updates["V"].append(V)
#     updates["previous_dV_dt"].append(dV_dt)
#     updates["P_pl"].append(P_pl)
#     updates["alpha"].append(alpha)
#     updates["Vflow_LA"].append(Vflow_LA)
#
#     # updates["t_eval6"] = updates["t_eval6"][1:]
#
#     return [dV_dt, d_alpha_dt]






# import numpy as np
# from collections import deque
#
# def cardiovascular_controller(t, state, params, time_history, exp_inputs, heart_inputs, resp_control_inputs, gas_exchange_inputs, updates, all_time, num_removed):
#     """
#     Afferent Pathways state variables:
#     theta_change_O2_sp, theta_change_CO2_sp, theta_change_O2_sv, theta_change_CO2_sv,
#     theta_change_O2_sh, theta_change_CO2_sh, P_tilda, f_ac, f_ap
#
#     Effectors for reflex control state variables:
#     R_ep_change, R_sp_change, R_rmp_n_change, R_amp_n_change, Vu_ev_change, Vu_sv_change, Vu_rmv_change,
#     Vu_amv_change, Emax_lv_change, Emax_rv_change, Ts_change, Tv_change
#
#     Blood Flow Local Control state variables:
#     xb_O2, xb_CO2, xh_O2, xh_CO2, Wh, xrm_O2, xrm_CO2, xam_O2, xM, x_met
#
#     """
#     (theta_change_O2_sp, theta_change_CO2_sp, theta_change_O2_sv, theta_change_CO2_sv, theta_change_O2_sh,
#      theta_change_CO2_sh, P_tilda, f_ac, f_ap, R_ep_change, R_sp_change,
#      R_rmp_n_change, R_amp_n_change, Vu_ev_change, Vu_sv_change, Vu_rmv_change, Vu_amv_change, Emax_lv_change,
#      Emax_rv_change, Ts_change, Tv_change, xb_O2, xb_CO2, xh_O2, xh_CO2, Wh, xrm_O2, xrm_CO2, xam_O2, xM, x_met) = state
#
#     ## Metabolic regulation
#     # constant parameters
#     AT = params["AT"]
#     MRTCO2_basal = params["MRTCO2_basal"]
#
#     # Other inputs
#     MRTCO2 = gas_exchange_inputs["MRTCO2"][-1]
#     # T_resp = 1 / resp_control_inputs["BF"]
#     previous_VE = exp_inputs["previous_VE"][-1]
#
#     a = updates["VE_integral"]
#     VE_integral = resp_control_inputs["VE_integral"][-1]
#
#
#     I = (MRTCO2 - MRTCO2_basal)/(AT - MRTCO2_basal)
#
#
#
#     ## Respiratory neuromuscular drive
#     # if t < TI:
#     #     RR = 1
#     # elif TI <= t < T_resp:
#     #     RR = 0, no need as Nt 0 outside of TI
#
#     # Nt = VE_integral
#
#     a0, a1, a2, tau, t1, t2 = exp_inputs["Nd"][-6:]
#     prev_flat_bit = updates["prev_flat_bit"][-1]
#
#     if t % (t1 + t2) < t1:
#         Nt = VE_integral - prev_flat_bit  # Take value minus previous flat bit
#     else:
#         Nt = 0  # Reset to zero
#         prev_flat_bit = VE_integral
#
#     ## CNS Ischemic Response
#     # constant parameters
#     g_ccsh = params["gccsh"]
#     g_ccsp = params["gccsp"]
#     g_ccsv = params["gccsv"]
#     kisc_sh = params["kisc_sh"]
#     kisc_sp = params["kisc_sp"]
#     kisc_sv = params["kisc_sv"]
#     PO2_sh = params["PO2_sh"]
#     PO2_sp = params["PO2_sp"]
#     PO2_sv = params["PO2_sv"]
#     tau_cc = params["tau_cc"]
#     tau_isc = params["tau_isc"]
#     theta_shn = params["theta_shn"]
#     theta_spn = params["theta_spn"]
#     theta_svn = params["theta_svn"]
#     x_sh = params["x_sh"]
#     x_sp = params["x_sp"]
#     x_sv = params["x_sv"]
#
#     PaCO2_n = params["PaCO2_n"]
#
#     # Other inputs
#     Pa_O2 = gas_exchange_inputs["Pa_O2"][-1]
#     Pa_CO2 = gas_exchange_inputs["Pa_CO2"][-1]
#
#     # cns response
#     w_sp = x_sp / (1 + np.exp((Pa_O2 - PO2_sp)/kisc_sp))
#     dtheta_change_O2_sp_dt = (-theta_change_O2_sp + w_sp) / tau_isc
#     dtheta_change_CO2_sp_dt = (-theta_change_CO2_sp + g_ccsp * (Pa_CO2 - PaCO2_n))/tau_cc
#
#     theta_sp = theta_spn - theta_change_O2_sp - theta_change_CO2_sp
#
#     w_sv = x_sv / (1 + np.exp((Pa_O2 - PO2_sv) / kisc_sv))
#     dtheta_change_O2_sv_dt = (-theta_change_O2_sv + w_sv) / tau_isc
#     dtheta_change_CO2_sv_dt = (-theta_change_CO2_sv + g_ccsv * (Pa_CO2 - PaCO2_n)) / tau_cc
#
#     theta_sv = theta_svn - theta_change_O2_sv - theta_change_CO2_sv
#
#     w_sh = x_sh / (1 + np.exp((Pa_O2 - PO2_sh) / kisc_sh))
#     dtheta_change_O2_sh_dt = (-theta_change_O2_sh + w_sh) / tau_isc
#     dtheta_change_CO2_sh_dt = (-theta_change_CO2_sh + g_ccsh * (Pa_CO2 - PaCO2_n)) / tau_cc
#
#     theta_sh = theta_shn - theta_change_O2_sh - theta_change_CO2_sh
#
#
#
#     ## Afferent Pathways
#     # afferent baroreflex constant parameters
#     f_ab_max = params["f_ab_max"]
#     f_ab_min = params["f_ab_min"]
#     k_ab = params["k_ab"]
#     P_n = params["P_n"]
#     tau_p = params["tau_p"]
#     tau_z = params["tau_z"]
#
#     # Other inputs
#     P_sa = heart_inputs["P_sa"][-2] # cardiovascular controller was run after cardio was run with states appended/updated so it must take the nonupdated version
#     dP_sa_dt = heart_inputs["dP_sa_dt"][-2]
#
#     f_ab = (f_ab_min + f_ab_max * np.exp((P_tilda - P_n)/k_ab)) / (1 + np.exp((P_tilda - P_n)/k_ab))
#     dP_tilda_dt = (P_sa + tau_z * dP_sa_dt - P_tilda) / tau_p
#
#     # afferent chemoreflex pathway constant parameters
#     f_ac_IC = params["f_ac_IC"]
#     f_acCO2_n = params["f_acCO2_n"]
#     f_ac_max = params["f_ac_max"]
#     f_ac_min = params["f_ac_min"]
#     k_ac = params["k_ac"]
#     K_H = params["K_H"]
#     PaO2_ac_n = params["PaO2_ac_n"]
#     PaCO2_n = params["PaCO2_n"]
#     tau_ac = params["tau_ac"]
#
#     if Pa_O2 >= 80:
#         K = K_H
#     elif 40 <= Pa_O2 < 80:
#         K = K_H - (1.2 * (Pa_O2 - 80) / 30)
#     else:
#         K = K_H - 1.6
#
#     phi_ac = ((f_ac_max + f_ac_min * np.exp((Pa_O2 - PaO2_ac_n)/k_ac))/(1 + np.exp((Pa_O2 - PaO2_ac_n)/k_ac)) *
#               (K * np.log(Pa_CO2/PaCO2_n) + f_acCO2_n))
#
#     d_fac_dt = (phi_ac - f_ac) / tau_ac
#
#     # afferent activity from Pulmonary Stretch Receptors constant parameters
#     f_ap_IC = params["f_ap_IC"]
#     G_ap = params["G_ap"]
#     tau_ap = params["tau_ap"]
#
#     # Other inputs
#     VT = resp_control_inputs["VT"][-1]
#
#     phi_ap = G_ap * VT
#     df_ap_dt = (phi_ap - f_ap)/tau_ap
#
#
#     ## Efferent Pathways constant parameters
#     (fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv, Io_v, kcc_sh, kcc_sp, kcc_sv,
#         kcc_v, Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v, Wb_sh, Wb_sp, Wb_sv, Wc_sh,
#         Wc_sp, Wc_sv, Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v) = [params[key] for key in
#                                                   ["fab_o", "fes_o", "fes_inf", "fes_max", "fev_o", "fev_inf", "kes", "kev", "Io_sh", "Io_sp", "Io_sv", "Io_v",
#         "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v", "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min", "Ysv_max", "Ysv_min",
#         "Yv_max", "Yv_min", "theta_v", "Wb_sh", "Wb_sp", "Wb_sv", "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v", "Wp_sh",
#         "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v"]]
#
#
#     Y_sh = (Ysh_min + Ysh_max * np.exp((I - Io_sh)/kcc_sh)) / (1 + np.exp((I - Io_sh)/kcc_sh))
#     f_ash = Wt_sh * Nt + Wb_sh * f_ab + Wc_sh * f_ac + Wp_sh * f_ap - theta_sh
#     f_sh = fes_inf + (fes_o - fes_inf) * np.exp(kes * f_ash) + Y_sh
#     if f_sh > fes_max:
#         f_sh = fes_max
#
#     Y_sp = (Ysp_min + Ysp_max * np.exp((I - Io_sp) / kcc_sp)) / (1 + np.exp((I - Io_sp) / kcc_sp))
#     f_asp = Wt_sp * Nt + Wb_sp * f_ab + Wc_sp * f_ac + Wp_sp * f_ap - theta_sp
#     f_sp = fes_inf + (fes_o - fes_inf) * np.exp(kes * f_asp) + Y_sp
#     if f_sp > fes_max:
#         f_sp = fes_max
#
#     Y_sv = (Ysv_min + Ysv_max * np.exp((I - Io_sv) / kcc_sv)) / (1 + np.exp((I - Io_sv) / kcc_sv))
#     f_asv = Wt_sv * Nt + Wb_sv * f_ab + Wc_sv * f_ac + Wp_sv * f_ap - theta_sv
#     f_sv = fes_inf + (fes_o - fes_inf) * np.exp(kes * f_asv) + Y_sv
#     if f_sv > fes_max:
#         f_sv = fes_max
#
#     Y_v = (Yv_min + Yv_max * np.exp((I - Io_v) / kcc_v)) / (1 + np.exp((I - Io_v) / kcc_v))
#     first_term = ((fev_o + fev_inf * np.exp((f_ab - fab_o)/kev)) / (1 + np.exp((f_ab - fab_o)/kev)))
#     # f_v = first_term - Wt_v * Nt - Wc_v * f_ac - Wp_v * f_ap - theta_v + Y_v
#     f_v = first_term - Wt_v * Nt + Wc_v * f_ac + Wp_v * f_ap - theta_v + Y_v
#     #
#     ## Effectors for reflex control
#     # resistances, unstressed volumes, and cardiac elastances.
#     # DEmax_lv = params["DEmax_lv"]
#     # DEmax_rv = params["DEmax_rv"]
#     # DR_amp = params["DR_amp"]
#     # DR_ep = params["DR_ep"]
#     # DR_rmp = params["DR_rmp"]
#     # DR_sp = params["DR_sp"]
#     # DV_amv = params["DV_amv"]
#     # DV_ev = params["DV_ev"]
#     # DV_rmv = params["DV_rmv"]
#     # DV_sv = params["DV_sv"]
#
#     (Emax_lv0, Emax_rv0, fes_min, GEmax_lv, GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0,
#      R_ep0, R_rmp0, R_sp0, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev, tau_Vrmv,
#      tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0) = [params[key] for key in
#         ["Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
#          "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp",
#          "tau_Rep", "tau_Rrmp", "tau_Rsp", "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0", "Vu_ev0", "Vu_rmv0",
#          "Vu_sv0"]]
#
#     f_sp_history, f_sh_history, f_v_history, f_sv_history, phi_met_history = [exp_inputs[key] for key in
#                                                                               ["f_sp_history", "f_sh_history",
#                                                                                "f_v_history", "f_sv_history",
#                                                                                "phi_met_history"]]
#
#     f_sp_history = deque(f_sp_history)
#     f_sh_history = deque(f_sh_history)
#     f_v_history = deque(f_v_history)
#     f_sv_history = deque(f_sv_history)
#     phi_met_history = deque(phi_met_history)
#
#     # added the below to get f_sp_delay from previous iterations.
#     delay_time2 = t - 2
#     if delay_time2 >= 0:
#         # Find the index for delay_time in time_history
#         f_sp_delay2 = f_sp_history[0]
#         f_sh_delay2 = f_sh_history[0]
#         f_sp_history.popleft()
#         f_sh_history.popleft()
#     else:
#         if t == 0:
#             f_sp_delay2 = f_sp
#             f_sh_delay2 = f_sh
#         else:
#             f_sp_delay2 = np.mean(f_sp_history)
#             f_sh_delay2 = 3.8576 #(f_shIC)
#
#     delay_time5 = t - 5
#     if delay_time5 >= 0:
#         # Find the index for delay_time in time_history
#         f_sv_delay5 = f_sv_history[0]
#         f_sv_history.popleft()
#     else:
#         if t == 0:
#             f_sv_delay5 = f_sv
#         else:
#             f_sv_delay5 = np.mean(f_sv_history)
#
#     # continue with equations
#     if f_sp < fes_min:
#         sigma_Rep = 0
#         sigma_Rsp = 0
#         sigma_Rrmp_n = 0
#         sigma_Ramp_n = 0
#
#     else:
#         sigma_Rep = GR_ep * np.log(f_sp_delay2 - fes_min + 1)
#         sigma_Rsp = GR_sp * np.log(f_sp_delay2 - fes_min + 1)
#         sigma_Rrmp_n = GR_rmp * np.log(f_sp_delay2 - fes_min + 1)
#         sigma_Ramp_n = GR_amp * np.log(f_sp_delay2 - fes_min + 1)
#
#     if f_sv < fes_min:
#         sigma_Vu_ev = 0
#         sigma_Vu_sv = 0
#         sigma_Vu_rmv = 0
#         sigma_Vu_amv = 0
#     else:
#         sigma_Vu_ev = GV_ev * np.log(f_sv_delay5 - fes_min + 1)
#         sigma_Vu_sv = GV_sv * np.log(f_sv_delay5 - fes_min + 1)
#         sigma_Vu_rmv = GV_rmv * np.log(f_sv_delay5 - fes_min + 1)
#         sigma_Vu_amv = GV_amv * np.log(f_sv_delay5 - fes_min + 1)
#
#     if f_sh < fes_min:
#         sigma_Emax_lv = 0
#         sigma_Emax_rv = 0
#     else:
#         sigma_Emax_lv = GEmax_lv * np.log(f_sh_delay2 - fes_min + 1)
#         sigma_Emax_rv = GEmax_rv * np.log(f_sh_delay2 - fes_min + 1)
#
#     dR_ep_change_dt = (- R_ep_change + sigma_Rep) / tau_Rep
#     dR_sp_change_dt = (- R_sp_change + sigma_Rsp) / tau_Rsp
#     dR_rmp_n_change_dt = (- R_rmp_n_change + sigma_Rrmp_n) / tau_Rrmp
#     dR_amp_n_change_dt = (- R_amp_n_change + sigma_Ramp_n) / tau_Ramp
#
#     dVu_ev_change_dt = (- Vu_ev_change + sigma_Vu_ev) / tau_Vev
#     dVu_sv_change_dt = (- Vu_sv_change + sigma_Vu_sv) / tau_Vsv
#     dVu_rmv_change_dt = (- Vu_rmv_change + sigma_Vu_rmv) / tau_Vrmv
#     dVu_amv_change_dt = (- Vu_amv_change + sigma_Vu_amv) / tau_Vamv
#
#     # dVu_ev_change_dt = 0
#     # dVu_sv_change_dt = 0
#     # dVu_rmv_change_dt = 0
#     # dVu_amv_change_dt = 0
#
#     dEmax_lv_change_dt = (- Emax_lv_change + sigma_Emax_lv) / tau_Emax_lv
#     dEmax_rv_change_dt = (- Emax_rv_change + sigma_Emax_rv) / tau_Emax_rv
#
#     R_ep = R_ep_change + R_ep0
#     R_sp = R_sp_change + R_sp0
#     R_rmp_n = R_rmp_n_change + R_rmp0
#     R_amp_n = R_amp_n_change + R_amp0
#
#     Vu_ev = Vu_ev_change + Vu_ev0
#     Vu_sv = Vu_sv_change + Vu_sv0
#     Vu_rmv = Vu_rmv_change + Vu_rmv0
#     Vu_amv = Vu_amv_change + Vu_amv0
#
#     Emax_lv = Emax_lv_change + Emax_lv0
#     Emax_rv = Emax_rv_change + Emax_rv0
#
#     if t>2:
#         a = 2
#
#     # heart period constants
#     DT_s = params["DT_s"]
#     DT_v = params["DT_v"]
#     fsh_IC = params["fsh_IC"]
#     fv_IC = params["fv_IC"]
#     GT_s = params["GT_s"]
#     GT_v = params["GT_v"]
#     T0 = params["T0"]
#     tau_Ts = params["tau_Ts"]
#     tau_Tv = params["tau_Tv"]
#
#     delay_time0_2 = t - DT_v
#     if delay_time0_2 >= 0:
#         # Find the index for delay_time in time_history
#         f_v_delay0_2 = f_v_history[0]
#         f_v_history.popleft()
#     else:
#         if t == 0:
#             f_v_delay0_2 = f_v
#         else:
#             f_v_delay0_2 = 4.2748 # np.mean(f_v_history), f_v_IC
#
#     if f_sh < fes_min:
#         sigma_Ts = 0
#     else:
#         sigma_Ts = GT_s * np.log(f_sh_delay2 - fes_min + 1)
#
#     d_Ts_change_dt = (- Ts_change + sigma_Ts) / tau_Ts
#     # d_Ts_change_dt = 0
#
#     sigma_Tv = GT_v * f_v_delay0_2
#     d_Tv_change_dt = (- Tv_change + sigma_Tv) / tau_Tv
#     # d_Tv_change_dt = 0
#
#     T = Tv_change + Ts_change + T0
#
#     HR = 1 / T
#
#
#
#     ## Blood Flow Local Control
#     # Cerebral Blood Flow constant parameters
#     A = params["A"]
#     B = params["B"]
#     C = params["C"]
#     D = params["D"]
#     Cvb_O2_n = params["Cvb_O2_n"]
#     gb_O2 = params["gb_O2"]
#     MO2_bp = params["MO2_bp"]
#     R_bpn = params["R_bpn"]
#     tau_CO2 = params["tau_CO2"]
#     tau_O2 = params["tau_O2"]
#
#     # other inputs
#     Ca_O2 = gas_exchange_inputs["Ca_O2"][-1]
#     Q_bp = heart_inputs["Q_bp"][-2]
#     Q_hp = heart_inputs["Q_hp"][-2]
#     Q_rmp = heart_inputs["Q_rmp"][-2]
#     Q_amp = heart_inputs["Q_amp"][-2]
#
#     G_bp = (1 / R_bpn) * (1 + xb_O2 + xb_CO2)
#     R_bp = 1 / G_bp
#     Cvb_O2 = Ca_O2 - MO2_bp / Q_bp
#
#     dxb_O2_dt = - (xb_O2 - gb_O2 * (Cvb_O2 - Cvb_O2_n)) / tau_O2
#
#     numerator = A + (B / (1 + C * np.exp(D * np.log10(Pa_CO2))))
#     denominator = A + (B / (1 + C * np.exp(D * np.log10(PaCO2_n))))
#     phi_b = numerator / denominator - 1
#
#     dxb_CO2_dt = (- xb_CO2 - phi_b) / tau_CO2
#
#     # Coronary and Resting Muscle Blood Flow constant parameters
#     Cvh_O2_n = params["Cvh_O2_n"]
#     Cvrm_O2_n = params["Cvrm_O2_n"]
#     gh_O2 = params["gh_O2"]
#     grm_O2 = params["grm_O2"]
#     Kh_CO2 = params["Kh_CO2"]
#     Krm_CO2 = params["Krm_CO2"]
#     MO2_hpn = params["MO2_hpn"]
#     MO2_rmp = params["MO2_rmp"]
#     R_hpn = params["R_hpn"]
#     tau_w = params["tau_w"]
#     W_hn = params["W_hn"]
#
#     # other inputs
#     Wh_lv = heart_inputs["Wh_lv"][-2]
#     Wh_rv = heart_inputs["Wh_rv"][-2]
#
#     # coronary
#     R_hp = R_hpn * (1 + xh_CO2) / (1 + xh_O2)
#
#     MO2_hp = MO2_hpn * Wh / W_hn
#     Cvh_O2 = Ca_O2 - MO2_hp / Q_hp
#
#     dxh_O2_dt = (- xh_O2 - gh_O2 * (Cvh_O2 - Cvh_O2_n)) / tau_O2
#
#     phi_h = (1 - np.exp((Pa_CO2 - PaCO2_n) / Kh_CO2)) / (1 + np.exp((Pa_CO2 - PaCO2_n) / Kh_CO2))
#
#     dxh_CO2_dt = (- xh_CO2 + phi_h) / tau_CO2
#
#     wh = Wh_lv + Wh_rv
#
#     dWh_dt = (wh - Wh) / tau_w
#
#     # resting muscle
#     R_rmp = R_rmp_n * (1 + xrm_CO2) / (1 + xrm_O2)
#     Cvrm_O2 = Ca_O2 - MO2_rmp / Q_rmp
#
#     dxrm_O2_dt = (- xrm_O2 - grm_O2 * (Cvrm_O2 - Cvrm_O2_n)) / tau_O2
#
#     phi_rm = (1 - np.exp((Pa_CO2 - PaCO2_n) / Krm_CO2)) / (1 + np.exp((Pa_CO2 - PaCO2_n) / Krm_CO2))
#
#     dxrm_CO2_dt = (- xrm_CO2 + phi_rm) / tau_CO2
#
#     # active muscle blood flow
#     Cvam_O2_n = params["Cvam_O2_n"]
#     Dmet = params["Dmet"]
#     gam_O2 = params["gam_O2"]
#     gM = params["gM"]
#     Io_met = params["Io_met"]
#     kmet = params["kmet"]
#     MO2_ampn = params["MO2_ampn"]
#     phi_max = params["phi_max"]
#     phi_min = params["phi_min"]
#     tau_M = params["tau_M"]
#     tau_met = params["tau_met"]
#
#     R_amp = R_amp_n / (1 + xam_O2 + x_met)
#
#     MO2_amp = MO2_ampn * (1 + xM)
#     Cvam_O2 = Ca_O2 - MO2_amp / Q_amp
#
#     dxam_O2_dt = (- xam_O2 - gam_O2 * (Cvam_O2 - Cvam_O2_n)) / tau_O2
#
#     dxM_dt = (- xM + gM * I) / tau_M
#
#     phi_met = (phi_min + phi_max * np.exp((I - Io_met) / kmet)) / (1 + np.exp((I - Io_met) / kmet))
#
#     delay_time_met = t - Dmet
#     if delay_time_met >= 0:
#         # Find the index for delay_time in time_history
#         phi_met_delay = phi_met_history[0]
#         phi_met_history.popleft()
#     else:
#         phi_met_delay = phi_met
#
#     dx_met_dt = (- x_met + phi_met_delay) / tau_met
#
#     if t != 0:
#         if t < all_time[-1]:
#             f_sh_history.append(f_sh_delay2)
#             f_sp_history.append(f_sp_delay2)
#             f_v_history.appendleft(f_v_delay0_2)
#             f_sv_history.appendleft(f_sv_delay5)
#             phi_met_history.appendleft(phi_met_delay)
#             for key in [
#                 "f_sp_history", "f_sh_history", "f_v_history", "phi_met_history", "f_sv_history",
#                 "Vu_ev", "Vu_amv", "Vu_rmv", "Vu_sv", "R_ep", "R_amp", "R_rmp", "R_sp", "R_bp", "R_hp", "HR",
#                 "Emax_lv", "Emax_rv", "I", "phi_met", "Nt", "Vu_sv_change", "prev_flat_bit", "Pa_O2"
#             ]:
#                 updates[key] = updates[key][:-num_removed]
#
#     # t_eval = updates["t_eval2"][0]
#     # check2 = t
#     # check3 = np.abs(t - t_eval)
#     #
#     # tolerance = 1e-3
#     # if np.abs(t - t_eval) < tolerance:
#         # update history
#     updates["f_sp_history"].append(f_sp)
#     updates["f_sh_history"].append(f_sh)
#     updates["f_v_history"].append(f_v)
#     updates["phi_met_history"].append(phi_met)
#     updates["f_sv_history"].append(f_sv)
#
#     updates["Vu_ev"].append(Vu_ev)
#     updates["Vu_amv"].append(Vu_amv)
#     updates["Vu_rmv"].append(Vu_rmv)
#     updates["Vu_sv"].append(Vu_sv)
#     updates["R_ep"].append(R_ep)
#     updates["R_amp"].append(R_amp)
#     updates["R_rmp"].append(R_rmp)
#     updates["R_sp"].append(R_sp)
#     updates["R_bp"].append(R_bp)
#     updates["R_hp"].append(R_hp)
#     updates["HR"].append(HR)
#     updates["Emax_lv"].append(Emax_lv)
#     updates["Emax_rv"].append(Emax_rv)
#     updates["I"].append(I)
#     updates["phi_met"].append(phi_met)
#     updates["Nt"].append(Nt)
#     updates["Vu_sv_change"].append(Vu_sv_change)
#     updates["prev_flat_bit"].append(prev_flat_bit)
#     updates["Pa_O2"].append(Pa_O2)
#
#     # updates["t_eval2"] = updates["t_eval2"][1:]
#
#     return [dtheta_change_O2_sp_dt, dtheta_change_CO2_sp_dt, dtheta_change_O2_sv_dt, dtheta_change_CO2_sv_dt,
#             dtheta_change_O2_sh_dt, dtheta_change_CO2_sh_dt, dP_tilda_dt, d_fac_dt, df_ap_dt, dR_ep_change_dt,
#             dR_sp_change_dt, dR_rmp_n_change_dt, dR_amp_n_change_dt, dVu_ev_change_dt, dVu_sv_change_dt,
#             dVu_rmv_change_dt, dVu_amv_change_dt, dEmax_lv_change_dt, dEmax_rv_change_dt, d_Ts_change_dt,
#             d_Tv_change_dt, dxb_O2_dt, dxb_CO2_dt, dxh_O2_dt, dxh_CO2_dt, dWh_dt, dxrm_O2_dt, dxrm_CO2_dt, dxam_O2_dt,
#             dxM_dt, dx_met_dt]
