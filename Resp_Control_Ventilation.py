import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
from collections import deque

from Gas_Exchange import gas_exchange
from Resp_Control_Breath_Optimiser import objective, calculate_P_musc_dP_dt, calculate_V_dV_dt


def resp_control_vent(t, state, params, updates, gas_exchange_inputs, num_removed, i, t_start):
    """
        Ventilation controller: Calculate VD, VD_flow, VE_flow, BF, TI
        Breathing pattern optimiser state variables: another function

        Other inputs: Gas exchange: Pa_CO2, Pa_O2, PbCO2, MRV
                      exp inputs: step_size, a0, a1, a2, tau, t1, t2

    """

    (VE_integral) = state[0].item()

    GV_dead = params["GV_dead"]
    Kbg = params["Kbg"]
    KcCO2 = params["KcCO2"]
    KcMRV = params["KcMRV"]
    KpCO2 = params["KpCO2"]
    KpO2 = params["KpO2"]
    V0_dead = params["V0_dead"]
    VA_rest = params["VA_rest"]
    lambda1 = params["lambda1"]
    lambda2 = params["lambda2"]
    n = params["n"]
    Pmax = params["Pmax"]
    Pmax_dot = params["Pmax_dot"]
    E_rs = params["E_rs"]
    R_rs = params["R_rs"]
    P_ao = params["P_ao"]

    if t == t_start:
        gas_exchange_index = i - num_removed
    else:
        gas_exchange_index = i - 1 - num_removed

    MRV = gas_exchange_inputs["MRV"][gas_exchange_index]


    a1, a2, tau, t1, t2 = updates["Nd"][-5:]
    Pa_O2_history = updates["Pa_O2_history"]
    Pa_CO2_history = updates["Pa_CO2_history"]
    Pb_CO2_history = updates["Pb_CO2_history"]



    last_breath_time = max(0, (t - updates["finish_breath_time"][-1]))
    A = updates["PamO2"]

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
            if resp_cycle < updates["resp_cycle"][i-1] and (updates["resp_cycle"][i-1] - resp_cycle) > 1 and num_removed == 0: # restarts
                t_start = updates["finish_breath_time"][-1]

                PamO2 = np.mean([val for (t_val, val) in updates["Pa_O2_history"] if t_start < t_val <= t])
                PamCO2 = np.mean([val for (t_val, val) in updates["Pa_CO2_history"] if t_start < t_val <= t])
                PmbCO2 = np.mean([val for (t_val, val) in updates["Pb_CO2_history"] if t_start < t_val <= t])

                updates["PamO2"].append(PamO2)
                updates["PamCO2"].append(PamCO2)
                updates["PmbCO2"].append(PmbCO2)

                min_time = updates["finish_breath_time"][-2]  # safe cutoff
                updates["Pa_O2_history"] = [(t, val) for (t, val) in updates["Pa_O2_history"] if t >= min_time]
                updates["Pa_CO2_history"] = [(t, val) for (t, val) in updates["Pa_CO2_history"] if t >= min_time]
                updates["Pb_CO2_history"] = [(t, val) for (t, val) in updates["Pb_CO2_history"] if t >= min_time]


    if PamO2 < 104:
        G3 = KpO2 * ((104 - PamO2) ** 4.9)
    else:
        G3 = 0

    VAflow = VA_rest * (KpCO2 * PamCO2 + KcCO2 * PmbCO2 + G3 + KcMRV * MRV - Kbg)

    VD = GV_dead * VAflow + V0_dead

    # Vt1 = VAflow * (t1 + t2) + VD
    dt = 0.001
    updates["dt"] = dt

    if t != t_start or t == 0:
        if resp_cycle < updates["resp_cycle"][i-1] and (updates["resp_cycle"][i-1] - resp_cycle) > 1 and num_removed == 0:

            bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2] bounds
            tolerance = 0.001

            required_params = [lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao]
            initial_guess = updates["Nd"][-2:]
            result = minimize(objective, initial_guess, args=(required_params, VAflow, VD, dt, tolerance), method='COBYLA', bounds=bounds)

            a2 = (-P_ao - E_rs * VAflow * (result.x[0] + result.x[1]) - E_rs * VD) / (result.x[0] ** 2) # VAflow constraint
            a1 = -2 * a2 * result.x[0] # dP_dt = 0 at t1
            Pt1 = a1 * result.x[0] + a2 * (result.x[0] ** 2)
            tau = t2 / (-np.log(tolerance/Pt1))

            updates["Nd"].append(a1)
            updates["Nd"].append(a2)
            updates["Nd"].append(tau)
            updates["Nd"].extend(result.x)


            t1, t2 = updates["Nd"][-2:]
            n_steps = int(np.round((t1 + t2) / dt)) + 1
            current_times = np.linspace(0, (t1 + t2), n_steps)
            updates["current_times"] = current_times

            P_for_current_breath, dP_dt_for_current_breath = calculate_P_musc_dP_dt(current_times, updates["Nd"][-2:], VAflow, VD, tolerance, E_rs, R_rs, P_ao)
            V_for_current_breath, dV_dt_for_current_breath = calculate_V_dV_dt(current_times, updates["Nd"][-2:], VAflow, VD, tolerance, E_rs, R_rs, P_ao)

            updates["P_musc_current"] = P_for_current_breath
            updates["V_current"] = V_for_current_breath
            updates["dV_dt_current"] = dV_dt_for_current_breath
            updates["dP_dt_current"] = dP_dt_for_current_breath
            updates["finish_breath_time"].append(t)

            # check optimisation results
            print(f"guess: {updates["Nd"][-5:]}")

    else:
        n_steps = int(np.round((t1 + t2) / dt)) + 1
        current_times = np.linspace(0, (t1 + t2), n_steps)
        updates["current_times"] = current_times


    BF = 1 / (t1 + t2)
    TI = t1
    VD_flow = BF * VD
    VE_flow = VAflow + VD_flow # in a second
    VT = VE_flow * (t1 + t2)


    # store ventilation variables
    last_breath_time = max(0, (t - updates["finish_breath_time"][-1]))


    resp_cycle = last_breath_time % (t1 + t2) # determine time within the breath

    V = np.interp(resp_cycle, updates["current_times"], updates["V_current"])
    dV_dt = np.interp(resp_cycle, updates["current_times"], updates["dV_dt_current"])
    P_musc = np.interp(resp_cycle, updates["current_times"], updates["P_musc_current"])

    # from cardiovascular controller
    if 0 <= (resp_cycle % (t1 + t2)) <= TI:
        d_VE_integral_dt = VE_flow
    else:
        d_VE_integral_dt = VE_flow  # doesn't matter if this is VE_flow or 0 as NT only considers inspiration

    if num_removed > 0:
        for key in [
            "BF", "TI", "VT", "VE_integral", "VD", "resp_cycle", "VAflow", "VE_flow", "P_musc", "dV_dt", "V"
        ]:
            updates[key][(i - num_removed): (i + 1)] = np.full((num_removed + 1,), 1e6)  # Replace values with 1e6
        i = i - num_removed

    # cardio inputs
    updates["BF"][i] = BF
    updates["TI"][i] = TI
    updates["VT"][i] = VT

    # cardio control inputs
    updates["VE_integral"][i] = VE_integral

    # gas inputs
    updates["VD"][i] = VD

    # resp control vent
    updates["resp_cycle"][i] = resp_cycle

    updates["VAflow"][i] = VAflow
    updates["VE_flow"][i] = VE_flow

    updates["P_musc"][i] = P_musc
    updates["dV_dt"][i] = dV_dt
    updates["V"][i] = V

    return [d_VE_integral_dt]