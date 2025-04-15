import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from Resp_Control_Breath_Optimiser import BreathOptimiser


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

    if t - updates["time_breath_history"][0] < 0:
        difference = 0
    else:
        difference = t - updates["time_breath_history"][0]

    resp_cycle = difference % (t1 + t2)
    if t <= (t1 + t2):
        PamO2 = np.mean(Pa_O2_history)
        PamCO2 = np.mean(Pa_CO2_history)
        PmbCO2 = np.mean(Pb_CO2_history)
        if np.isclose(resp_cycle, (t1 + t2), atol=3e-03, equal_nan=False) and updates["Pa_O2_history"]:
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

        if np.isclose(resp_cycle, (t1 + t2), atol=3e-03, equal_nan=False) and updates["Pa_O2_history"]: # restarts
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

    VAflow = VA_rest * (KpCO2 * PamCO2 + KcCO2 * PmbCO2 + G3 + KcMRV * MRV - Kbg)
    VD = GV_dead * VAflow + V0_dead

    # comment out to uncouple breath optimiser
    if len(updates["time_breath_history"]) > 1:
        A = updates["resp_cycle"]
        if resp_cycle < updates["resp_cycle"][i-1]  and (updates["resp_cycle"][i-1] - resp_cycle) > 1:
            # bounds = [(-20, 60), (-30, 10), (0.1, 1), (0.2, 5), (0.2, 5)] # [a1, a2, tau, t1, t2]
            bounds = [(-30, 10), (0.1, 2), (0.4, 4), (0.4, 4)]  # [a2, tau, t1, t2]
            times_array = np.array(updates["time_breath_history"]) - updates["time_breath_history"][0]

            opt = BreathOptimiser(params, times_array, np.diff(times_array), updates["time_breath_history"])

            # Define the nonlinear constraint using latest_volume
            # nlc1 = NonlinearConstraint(lambda x: x[0] + x[1] * x[3], lb=0, ub=np.inf)
            # nlc2 = NonlinearConstraint(lambda x: x[0] + 2 * x[1] * x[3], lb=-0.005, ub=0.005)

            nlc_a2 = NonlinearConstraint(lambda x: x[0], lb=-float('inf'), ub=0)
            nlc_V = NonlinearConstraint(lambda x: opt.constraint_function(x, VD = VD, VA = VAflow), lb=-0.01, ub=0.01)
            nlc_tau = NonlinearConstraint(lambda x: opt.tau_constraint(x), lb=-0.01, ub=0.01)
            T_total = times_array[-1]
            nlc_duration = NonlinearConstraint(lambda x: x[1] + x[2], lb=T_total - 0.5, ub=T_total + 0.5)


            constraints = [nlc_a2, nlc_V, nlc_tau, nlc_duration]

            # Optimize
            result = minimize(opt.objective, updates["Nd"][-4:], method='SLSQP', constraints=constraints, bounds=bounds)
            a1 = -2 * result.x[0] * result.x[2]
            updates["Nd"].append(a1)
            updates["Nd"].extend(result.x)
            updates["time_breath_history"].clear()
            updates["time_breath_history"].append(t)
            updates["J"].append(result.fun)
            a1, a2, tau, t1, t2 = updates["Nd"][-5:]
            print((a1 * t1 + a2 * (t1 ** 2)) * np.exp(-t2 / tau))

    
    a1, a2, tau, t1, t2 = updates["Nd"][-5:]
    BF = 1 / (t1 + t2)
    TI = t1
    VD_flow = BF * VD
    VE_flow = VAflow + VD_flow
    VT = VE_flow * (t1 + t2)

    # from cardiovascular controller
    A = difference % (t1 + t2)

    if t > 27.39:
        A = updates["time_breath_history"]

    if 0 <= (difference % (t1 + t2)) <= TI:
        # NT = VE_flow
        d_VE_integral_dt = VE_flow
    else:
        d_VE_integral_dt = VE_flow


    if num_removed > 0:
        for key in [
            "VE_integral", "VD", "BF", "TI", "VT", "VAflow", "VE_flow"
        ]:
            updates[key][(i - num_removed): (i + 1)] = np.full((num_removed + 1,), 1e6)  # Replace values with 1e6
        # for key in ["time_breath_history"]:
        #     del updates[key][-num_removed:]

        i = i - num_removed

    updates["VE_integral"][i] = VE_integral
    updates["VD"][i] = VD
    updates["BF"][i] = BF
    updates["TI"][i] = TI
    updates["VT"][i] = VT
    updates["VAflow"][i] = VAflow
    updates["VE_flow"][i] = VE_flow
    updates["difference"][i] = difference
    updates["resp_cycle"][i] = resp_cycle

    if np.isclose(t % 0.001, 0.0, atol=0.0005):
        if t != 0:
            updates["time_breath_history"].append(t)


    return [d_VE_integral_dt]

