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
