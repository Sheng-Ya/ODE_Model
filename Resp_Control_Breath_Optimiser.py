import numpy as np

global_results = {}

def breath_optimiser(initial_Nd_guess, t, time_history, params, exp_inputs, resp_mech_inputs, updates, all_time, num_removed):
    """
     Function to obtain a0, a1, a2, tau, t1, t2
     Other inputs: step_size, previous dV_dt, current dV_dt, P_musc, previous WI, previous WE

    """
    [a0, a1, a2, tau, t1, t2] = initial_Nd_guess

    # Breathing Pattern Optimiser
    lambda1 = params["lambda1"]
    lambda2 = params["lambda2"]
    n = params["n"]
    Pmax = params["Pmax"]
    Pmax_dot = params["Pmax_dot"]

    # other inputs
    dV_dt = resp_mech_inputs["dV_dt"][-1]
    previous_dV_dt = exp_inputs["previous_dV_dt"][-1]
    P_musc = resp_mech_inputs["P_musc"][-1]
    previous_WI = exp_inputs["previous_WI"][-1]
    previous_WE = exp_inputs["previous_WE"][-1]


    if (len(time_history) > 0): # After the first step
        step_size = t - time_history[-1]
        if (t - time_history[-1] == 0): # step if iterating at the same time
            step_size = time_history[-1]
    else: # first step
        step_size = t

    breath = t % (t1 + t2)

    if 0 <= breath <= t1:
        dP_musc_dt = a1 + 2 * a2 * t
    elif t1 < breath <= (t1 + t2):
        P_musc_t1 = a0 + a1 * t1 + a2 * (t1 ** 2)
        dP_musc_dt = P_musc_t1 * np.exp(-(t - t1) / tau) * (-1 / tau)

    d2V_dt2_squared = ((previous_dV_dt - dV_dt) / step_size) ** 2

    E1 = (1 - P_musc / Pmax) ** n
    E2 = (1 - dP_musc_dt / Pmax_dot) ** n

    WI = previous_WI
    WE = previous_WE

    if 0 <= breath <= t1:
        dWI_dt = (1/(t1+t2)) * (P_musc * dV_dt / (E1 * E2)) + lambda1 * d2V_dt2_squared
        WI = previous_WI + dWI_dt * step_size  # Integrate using Euler's method
    else:
        dWE_dt = (1/(t1+t2)) * d2V_dt2_squared
        WE = previous_WE + dWE_dt * step_size


    previous_WI = WI
    previous_WE = WE

    J = WI + lambda2 * WE

    # t_eval = updates["t_eval4"][0]
    # tolerance = 1e-3
    # if np.abs(t - t_eval) < tolerance:
        # Store WI and WE globally
    updates["WI"].append(WI)
    updates["WE"].append(WE)
    updates["previous_WI"].append(previous_WI)
    updates["previous_WE"].append(previous_WE)

    # updates["t_eval4"] = updates["t_eval4"][1:]

    # time_history.append(t)

    return J



