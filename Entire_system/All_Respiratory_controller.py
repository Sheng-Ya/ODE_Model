import numpy as np
from scipy.optimize import minimize
from Resp_Control_Breath_Optimiser import objective, calculate_P_musc_dP_dt, calculate_V_dV_dt


def resp_control_vent(t, state, params, updates, gas_exchange_inputs, num_removed, t_start, time_saved, i, BUFFER_LIMIT, Parameters):
    """
        Ventilation controller: Calculate VD, VD_flow, VE_flow, BF, TI
        Breathing pattern optimiser state variables: another function

        Other inputs: Gas exchange: Pa_CO2, Pa_O2, PbCO2, MRV
                      exp inputs: step_size, a0, a1, a2, tau, t1, t2

    """

    (VE_integral) = state[0].item()

    (GV_dead, Kbg, KcCO2, KcMRV, KpCO2, KpO2, V0_dead, VA_rest, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao) = \
        (params[k] if k in params else Parameters[k] for k in [
            "GV_dead", "Kbg", "KcCO2", "KcMRV", "KpCO2", "KpO2", "V0_dead", "VA_rest", "lambda1", "lambda2", "n",
            "Pmax", "Pmax_dot", "E_rs", "R_rs", "P_ao"
        ])

    if t == t_start:
        gas_exchange_index = i % BUFFER_LIMIT
    else:
        gas_exchange_index = (i - num_removed - 1) % BUFFER_LIMIT

    MRV = gas_exchange_inputs["MRV_store"][gas_exchange_index]

    a1, a2, tau, t1, t2 = updates["Nd"][-5:]
    Pa_O2_history = updates["Pa_O2_history"]
    Pa_CO2_history = updates["Pa_CO2_history"]
    Pb_CO2_history = updates["Pb_CO2_history"]

    last_breath_time = max(0, (t - updates["finish_breath_time"][-1]))

    resp_cycle = last_breath_time % (t1 + t2)
    if t <= (t1 + t2) and updates["finish_breath_time"][-1] == 0:
        PamO2 = Pa_O2_history[0][1]
        PamCO2 = Pa_CO2_history[0][1]
        PmbCO2 = Pb_CO2_history[0][1]
    else:
        PamO2 = updates["PamO2"][-1]
        PamCO2 = updates["PamCO2"][-1]
        PmbCO2 = updates["PmbCO2"][-1]

        if t != t_start:
            if resp_cycle < updates["resp_cycle_store"][(i - num_removed - 1) % BUFFER_LIMIT] and (
                    updates["resp_cycle_store"][
                        (i - num_removed - 1) % BUFFER_LIMIT] - resp_cycle) > 1 and num_removed == 0:  # restarts
                t_last_breath = updates["finish_breath_time"][-1]

                PamO2 = np.mean([val for (t_val, val) in updates["Pa_O2_history"] if t_last_breath < t_val <= t])
                PamCO2 = np.mean([val for (t_val, val) in updates["Pa_CO2_history"] if t_last_breath < t_val <= t])
                PmbCO2 = np.mean([val for (t_val, val) in updates["Pb_CO2_history"] if t_last_breath < t_val <= t])

                updates["PamO2"].append(PamO2)
                updates["PamCO2"].append(PamCO2)
                updates["PmbCO2"].append(PmbCO2)

                min_time = updates["finish_breath_time"][-2]
                updates["Pa_O2_history"] = [(t, val) for (t, val) in updates["Pa_O2_history"] if t >= min_time]
                updates["Pa_CO2_history"] = [(t, val) for (t, val) in updates["Pa_CO2_history"] if t >= min_time]
                updates["Pb_CO2_history"] = [(t, val) for (t, val) in updates["Pb_CO2_history"] if t >= min_time]

    if PamO2 < 104:
        G3 = KpO2 * ((104 - PamO2) ** 4.9)
    else:
        G3 = 0

    # VA_rest = 0.03

    # VAflow = VA_rest * (KcCO2 * (PmbCO2 - params["PbCO2IC"])) + KpCO2 * PamCO2 + G3 + KcMRV * MRV + VA_rest - 8.29
    VAflow = VA_rest * (KpCO2 * PamCO2 + KcCO2 * PmbCO2 + G3 + KcMRV * MRV - Kbg)
    # VAflow = VA_rest + VA_rest * (KcCO2 * (PmbCO2 - 44.87)) + G3 + KcMRV * MRV
    # VAflow = 0.0867
    # V0_dead = 0.13

    # central chemoreceptor
    # if PmbCO2 < 48.62:
    #     phi_Pmb = 0.2332 * (PmbCO2 - 43.613) * VA_rest
    # else:
    #     phi_Pmb = 0.3803 * (PmbCO2 - 43.613 - 0.551) * VA_rest
    #
    # dVc_dt  = (phi_Pmb - Vc) * (1/100) # tau_c = 100
    #
    # # peripheral chemoreceptor
    # CaO2 = gas_exchange_inputs["Ca_O2"][gas_exchange_index]
    # G_PamCO2 = -(5/60) * CaO2 * PamCO2 - (147/60) * CaO2 + (1.9/60) * PamCO2
    #
    # dVp_dt = (G_PamCO2 - Vp) * (1 / 10)  # tau_c = 100

    # VAflow = VA_rest + Vc
    # VAflow = VA_rest

    VD = GV_dead * VAflow + V0_dead

    # Vt1 = VAflow * (t1 + t2) + VD
    dt = 0.001
    updates["dt"] = dt

    if t != t_start or t == 0:
        if resp_cycle < updates["resp_cycle_store"][(i - num_removed - 1) % BUFFER_LIMIT] and (
                updates["resp_cycle_store"][
                    (i - num_removed - 1) % BUFFER_LIMIT] - resp_cycle) > 1 and num_removed == 0:
            bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2] bounds
            tolerance = 0.001

            # nlc_tau = NonlinearConstraint(lambda x: opt.tau_constraint(x), lb=-0, ub=0.5) # tau constraint: at time (t1 + t2), P_musc is 0
            # nlc_tau = {
            #     'type': 'ineq',
            #     'fun': opt.tau_constraint
            # }
            # Optimize
            # print((a1 * t1 + a2 * (t1 ** 2) - P_ao) - E_rs * (VAflow * (t1 + t2) + VD))
            # print((-a2 * (t1) ** 2 - P_ao - E_rs * VAflow * t1 - E_rs * VD) / (E_rs * VAflow))

            required_params = [lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao]
            initial_guess = updates["Nd"][-2:]
            result = minimize(objective, initial_guess, args=(required_params, VAflow, VD, dt, tolerance),
                              method='COBYLA', bounds=bounds)

            a2 = (-P_ao - E_rs * VAflow * (result.x[0] + result.x[1]) - E_rs * VD) / (
                        result.x[0] ** 2)  # VAflow constraint
            a1 = -2 * a2 * result.x[0]  # dP_dt = 0 at t1
            Pt1 = a1 * result.x[0] + a2 * (result.x[0] ** 2)
            tau = t2 / (-np.log(tolerance / Pt1))

            updates["Nd"].append(a1)
            updates["Nd"].append(a2)
            updates["Nd"].append(tau)
            updates["Nd"].extend(result.x)
            # updates["J"].append(result.fun)

            # if 910.5 > t > 910:
            #     a2 = (-P_ao - E_rs * VAflow * (0.5 + 0.8) - E_rs * VD) / (0.5 ** 2) # VAflow constraint
            #     a1 = -2 * a2 * 0.5  # dP_dt = 0 at t1
            #     Pt1 = a1 * 0.5 + a2 * (0.5 ** 2)
            #     tau = 0.8 / (-np.log(tolerance/Pt1))
            #     updates["Nd"].append(a1)
            #     updates["Nd"].append(a2)
            #     updates["Nd"].append(tau)
            #     updates["Nd"].append(0.5)
            #     updates["Nd"].append(0.8)

            t1, t2 = updates["Nd"][-2:]
            n_steps = int(np.round((t1 + t2) / dt)) + 1
            current_times = np.linspace(0, (t1 + t2), n_steps)
            updates["current_times"] = current_times

            P_for_current_breath, dP_dt_for_current_breath = calculate_P_musc_dP_dt(current_times, updates["Nd"][-2:],
                                                                                    VAflow, VD, tolerance, E_rs, R_rs,
                                                                                    P_ao)
            V_for_current_breath, dV_dt_for_current_breath = calculate_V_dV_dt(current_times, updates["Nd"][-2:],
                                                                               VAflow, VD, tolerance, E_rs, R_rs, P_ao)

            updates["P_musc_current"] = P_for_current_breath
            updates["V_current"] = V_for_current_breath
            updates["dV_dt_current"] = dV_dt_for_current_breath
            updates["dP_dt_current"] = dP_dt_for_current_breath
            updates["finish_breath_time"].append(t)

            # check optimisation results
            print(f"guess: {updates["Nd"][-5:]}")
            # check whether pressure at time t1+t2 is 0
            # print((a1 * t1 + a2 * (t1 ** 2)) * np.exp(-t2 / tau))
            # check whether dV_dt = 0 at t1
            # print((a1 * t1 + a2 * (t1 ** 2) - P_ao) - E_rs * (VAflow * (t1 + t2) + VD))
            # print(VAflow * (t1 + t2) + VD)
            # plt.plot(current_times, V_for_current_breath, label="V")
            # plt.plot(current_times, dV_dt_for_current_breath, label="dV_dt")
            # plt.plot(current_times, VA_for_current_breath, label="VA")

            # plt.legend()
            # plt.show()
    else:
        n_steps = int(np.round((t1 + t2) / dt)) + 1
        current_times = np.linspace(0, (t1 + t2), n_steps)
        updates["current_times"] = current_times

    BF = 1 / (t1 + t2)
    TI = t1
    VD_flow = BF * VD
    VE_flow = VAflow + VD_flow  # in a second
    VT = VE_flow * (t1 + t2)

    # store ventilation variables
    last_breath_time = max(0, (t - updates["finish_breath_time"][-1]))
    resp_cycle = last_breath_time % (t1 + t2)  # determine time within the breath

    V = np.interp(resp_cycle, updates["current_times"], updates["V_current"])
    dV_dt = np.interp(resp_cycle, updates["current_times"], updates["dV_dt_current"])
    P_musc = np.interp(resp_cycle, updates["current_times"], updates["P_musc_current"])

    # from cardiovascular controller
    if 0 <= (resp_cycle % (t1 + t2)) <= TI:
        d_VE_integral_dt = VE_flow
    else:
        d_VE_integral_dt = VE_flow  # doesn't matter if this is VE_flow or 0 as NT only considers inspiration

    # Combine keys and corresponding values for updates
    for key, new_value in zip(
            [  # Cardio inputs
                "BF_store", "TI_store", "VT_store",

                # Cardio control inputs
                "VE_integral_store",

                # Gas inputs
                "VD_store",

                # Resp control vent
                "resp_cycle_store", "VAflow_store", "VE_flow_store", "P_musc_store", "dV_dt_store", "V_store"],

            [  # Corresponding values
                BF, TI, VT, VE_integral, VD,
                resp_cycle, VAflow, VE_flow, P_musc, dV_dt, V]
    ):
        updates[key][((i - num_removed) % BUFFER_LIMIT)] = new_value

        # just for plotting purposes
    # if ((t % time_saved) < 0.001 or (time_saved - (t % time_saved)) < 0.001) and num_removed == 0:
    #     j = updates["j"].item()

    keys_and_values = zip(
        [  # Cardio inputs
            "BF", "TI", "VT",

            # Gas inputs
            "VD",

            # Resp control vent
            "VAflow", "VE_flow",
            "P_musc", "dV_dt", "V"],

        [  # Corresponding values
            BF, TI, VT, VD,
            VAflow, VE_flow, P_musc, dV_dt, V])

    for key, value in keys_and_values:
        updates[key][updates["j"].item() - num_removed] = value

        # updates["j"] = np.array(updates["j"].item() + 1)

    # keys_and_values = zip(
    #     [   # Cardio inputs
    #         "BF", "TI", "VT",
    #
    #         # Cardio control inputs
    #         "VE_integral",
    #
    #         # Gas inputs
    #         "VD",
    #
    #         # Resp control vent
    #         "resp_cycle", "VAflow", "VE_flow",
    #         "P_musc", "dV_dt", "V" ],
    #
    #     [   # Corresponding values
    #         BF, TI, VT, VE_integral, VD, resp_cycle,
    #         VAflow, VE_flow, P_musc, dV_dt, V])
    #
    # for key, value in keys_and_values:
    #     updates[key][updates["j"].item() - num_removed] = value

    return [d_VE_integral_dt]

