import numpy as np

def resp_control_vent(t, state, params, exp_inputs, gas_exchange_inputs, updates, num_removed, i):
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
        gas_exchange_index = i
    elif num_removed > 0:
        gas_exchange_index = i - num_removed - 1
    else:
        gas_exchange_index = i - 1

    MRV = gas_exchange_inputs["MRV"][gas_exchange_index]

    a0, a1, a2, tau, t1, t2 = exp_inputs["Nd"][:6]
    Pa_O2_history = exp_inputs["Pa_O2_history"]
    Pa_CO2_history = exp_inputs["Pa_CO2_history"]
    Pb_CO2_history = exp_inputs["Pb_CO2_history"]

    resp_cycle = t % (t1 + t2)
    if t <= (t1 + t2):
        PamO2 = np.mean(Pa_O2_history)
        PamCO2 = np.mean(Pa_CO2_history)
        PmbCO2 = np.mean(Pb_CO2_history)
        if np.isclose(resp_cycle, 1, atol=3e-03, equal_nan=False) and updates["Pa_O2_history"]:
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

        if np.isclose(resp_cycle, 1, atol=3e-03, equal_nan=False) and updates["Pa_O2_history"]: # restarts
            PamO2 = np.mean(Pa_O2_history)
            PamCO2 = np.mean(Pa_CO2_history)
            PmbCO2 = np.mean(Pb_CO2_history)

            updates["PamO2"].append(PamO2)
            updates["PamCO2"].append(PamCO2)
            updates["PmbCO2"].append(PmbCO2)

            exp_inputs["Pa_O2_history"].clear()
            exp_inputs["Pa_CO2_history"].clear()
            exp_inputs["Pb_CO2_history"].clear()


    # num_steps_per_cycle = int((t1 + t2) / step_size)  # Steps in one cycle
    # current_index = int(t / step_size)  # Index corresponding to time t
    # end = (current_index // num_steps_per_cycle) * num_steps_per_cycle  + 1 # End of the previous cycle
    # start = end - num_steps_per_cycle
    # PamO2 = np.mean(Pa_O2[start:end])
    # PamCO2 = np.mean(Pa_CO2[start:end])
    # PmbCO2 = np.mean(Pb_CO2[start:end])

    BF = 1 / (t1 + t2)

    TI = t1


    if PamO2 < 104:
        G3 = KpO2 * ((104 - PamO2) ** 4.9)
    else:
        G3 = 0

    VAflow = VA_rest * (KpCO2 * PamCO2 + KcCO2 * PmbCO2 + G3 + KcMRV * MRV - Kbg)

    # if VAflow < 0:
    #     VAflow = 0

    VD = GV_dead * VAflow + V0_dead
    VD_flow = BF * VD
    VE_flow = VAflow + VD_flow

    VT = VE_flow * (t1 + t2)

    if VT < 0:
        A = 2

    # from cardiovascular controller
    if 0 <= (t % (t1 + t2)) <= TI:
        # NT = VE_flow
        d_VE_integral_dt = VE_flow
    else:
        d_VE_integral_dt = VE_flow


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






    # bounds = [(0, None), (0, None), (0, None), (0, None), (0.1, None), (0.1, None)]

    # # Optimize
    # result = minimize(breath_optimiser, exp_inputs["Nd"][-6:], args=(t, time_history, params, Next_Conditions, Next_Conditions, Next_Conditions), method='SLSQP', bounds=bounds)
    # updates["Nd"].append(result.x)

    return [d_VE_integral_dt]

