import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint

from Resp_Control_Breath_Optimiser_just_resp import BreathOptimiser, simulate_euler


def resp_control_vent(t, state, params, updates, gas_exchange_inputs, num_removed, i):
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

    if t == 0:
        updates["time_breath_history"].append(0)
        gas_exchange_index = i
    elif num_removed > 0:
        gas_exchange_index = i - num_removed - 1
    else:
        gas_exchange_index = i - 1

    MRV = gas_exchange_inputs["MRV"][gas_exchange_index]

    a1, a2, tau, t1, t2 = updates["Nd"][-5:]
    Pa_O2_history = updates["Pa_O2_history"]
    Pa_CO2_history = updates["Pa_CO2_history"]
    Pb_CO2_history = updates["Pb_CO2_history"]

    A = updates["finish_breath_time"][-1]
    last_breath_time = max(0, t - updates["finish_breath_time"][-1])
    AA = updates["resp_cycle"][i-1]

    resp_cycle = last_breath_time % (t1 + t2)
    if t <= (t1 + t2) and updates["finish_breath_time"][-1] == 0:
        PamO2 = Pa_O2_history[0]
        PamCO2 = Pa_CO2_history[0]
        PmbCO2 = Pb_CO2_history[0]
        if resp_cycle < updates["resp_cycle"][i-1] and (updates["resp_cycle"][i-1] - resp_cycle) > 1:
            updates["PamO2"].append(PamO2)
            updates["PamCO2"].append(PamCO2)
            updates["PmbCO2"].append(PmbCO2)

            updates["Pa_O2_history"].clear()
            updates["Pa_CO2_history"].clear()
            updates["Pb_CO2_history"].clear()
    else:
        PamO2 = updates["PamO2"][-1]
        PamCO2 = updates["PamCO2"][-1]
        PmbCO2 = updates["PmbCO2"][-1]

        if resp_cycle < updates["resp_cycle"][i-1] and (updates["resp_cycle"][i-1] - resp_cycle) > 1: # restarts
            PamO2 = np.mean(Pa_O2_history)
            PamCO2 = np.mean(Pa_CO2_history)
            PmbCO2 = np.mean(Pb_CO2_history)

            updates["PamO2"].append(PamO2)
            updates["PamCO2"].append(PamCO2)
            updates["PmbCO2"].append(PmbCO2)

            updates["Pa_O2_history"].clear()
            updates["Pa_CO2_history"].clear()
            updates["Pb_CO2_history"].clear()


    if PamO2 < 104:
        G3 = KpO2 * ((104 - PamO2) ** 4.9)
    else:
        G3 = 0

    # VA_rest = 0.03
    AAA = params["PbCO2IC"]
    A = VA_rest * (KcCO2 * (PmbCO2 - params["PbCO2IC"])) #+ KpCO2 * PamCO2

    # VAflow = VA_rest * (KcCO2 * (PmbCO2 - params["PbCO2IC"])) + KpCO2 * PamCO2 + G3 + KcMRV * MRV + VA_rest - 8.29
    VAflow = VA_rest * (KpCO2 * PamCO2 + KcCO2 * PmbCO2 + G3 + KcMRV * MRV - Kbg)
    # VAflow = VA_rest + VA_rest * (KcCO2 * (PmbCO2 - 44.87)) + G3 + KcMRV * MRV
    VAflow = 0.0867
    V0_dead = 0.13

    VD = GV_dead * VAflow + V0_dead

    Vt1 = VAflow * (t1 + t2) + VD

    if resp_cycle < updates["resp_cycle"][i-1] and (updates["resp_cycle"][i-1] - resp_cycle) > 1:
        bounds = [(0.1, 1), (0.5, 3), (0.5, 3)]  # [tau, t1, t2] bounds
        dt = 0.001
        updates["dt"] = dt
        opt = BreathOptimiser(params, VAflow, VD, dt)

        nlc_tau = NonlinearConstraint(lambda x: opt.tau_constraint(x), lb=-0.01, ub=0.01) # tau constraint: at time (t1 + t2), P_musc is 0

        # Optimize
        # print((a1 * t1 + a2 * (t1 ** 2) - params["P_ao"]) - params["E_rs"] * (VAflow * (t1 + t2) + VD))
        # print((-a2 * (t1) ** 2 - params["P_ao"] - params["E_rs"] * VAflow * t1 - params["E_rs"] * VD) / (params["E_rs"] * VAflow))
        result = minimize(opt.objective, updates["Nd"][-3:], method='SLSQP', constraints=[nlc_tau], bounds=bounds)
        # result = minimize(opt.objective, np.array([-3, 0.4, 1.8]), method='SLSQP', constraints=constraints, bounds=bounds)

        a2 = (-params["P_ao"] - params["E_rs"] * VAflow * (result.x[1] + result.x[2]) - params["E_rs"] * VD) / (result.x[1] ** 2)
        a1 = -2 * a2 * result.x[1] # dP_dt = 0 at t1
        updates["Nd"].append(a1)
        updates["Nd"].append(a2)
        updates["Nd"].extend(result.x)
        updates["J"].append(result.fun)

        a1, a2, tau, t1, t2 = updates["Nd"][-5:]
        n_steps = int(np.round((t1 + t2) / dt)) + 1
        current_times = np.linspace(0, (t1 + t2), n_steps)
        print(VAflow, VD)
        print((a1 * t1 + a2 * (t1 ** 2) - params["P_ao"]) - params["E_rs"] * (VAflow * (t1 + t2) + VD))

        P_for_current_breath, dP_dt_for_current_breath = opt.calculate_P_musc_dP_dt(current_times, updates["Nd"][-3:])
        # V_for_current_breath, dV_dt_for_current_breath = simulate_euler(current_times, P_for_current_breath, params["R_rs"], params["P_ao"], params["E_rs"])
        V_for_current_breath, dV_dt_for_current_breath = opt.calculate_V_dV_dt(current_times, updates["Nd"][-3:])

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
        # print((a1 * t1 + a2 * (t1 ** 2) - params["P_ao"]) - params["E_rs"] * (VAflow * (t1 + t2) + VD))
        # print(VAflow * (t1 + t2) + VD)
        # plt.plot(current_times, V_for_current_breath, marker='o', label="V")
        # plt.plot(current_times, dV_dt_for_current_breath, marker='o', label="dV_dt")
        # plt.legend()
        # plt.show()

    BF = 1 / (t1 + t2)
    TI = t1
    VD_flow = BF * VD
    VE_flow = VAflow + VD_flow # in a second
    VT = VE_flow * (t1 + t2)

    # from cardiovascular controller
    if 0 <= (resp_cycle % (t1 + t2)) <= TI:
        d_VE_integral_dt = VE_flow
    else:
        d_VE_integral_dt = VE_flow # doesn't matter if this is VE_flow or 0 as NT only considers inspiration


    if num_removed > 0:
        for key in [
            "VE_integral", "VD", "BF", "TI", "VT", "VAflow", "VE_flow"
        ]:
            updates[key][(i - num_removed): (i + 1)] = np.full((num_removed + 1,), 1e6)  # Replace values with 1e6
        i = i - num_removed

    updates["VE_integral"][i] = VE_integral
    updates["VD"][i] = VD
    updates["BF"][i] = BF
    updates["TI"][i] = TI
    updates["VT"][i] = VT
    updates["VAflow"][i] = VAflow
    updates["VE_flow"][i] = VE_flow
    updates["resp_cycle"][i] = resp_cycle

    return [d_VE_integral_dt]

