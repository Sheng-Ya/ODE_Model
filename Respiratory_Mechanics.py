import numpy as np
from scipy.interpolate import interp1d

def respiratory_mechanics(t, state, params, updates, num_removed, i):
    """
        Pulmonary Mechanics state variables: V
        Upper Airway state variables: alpha

    """

    (Vflow_ua) = state[0].item()

    ## Pulmonary Mechanics
    E_CW = params["E_CW"]
    E_L = params["E_L"]
    E_rs = params["E_rs"]
    k_aw1 = params["k_aw1"]
    k_aw2 = params["k_aw2"]
    P_ao = params["P_ao"]
    R_rs = params["R_rs"]

    ## Upper Airways
    A0_ua = params["A0_ua"]
    b_ua = params["b_ua"]
    C_ua = params["C_ua"]
    K_ua = params["K_ua"]
    Pcrit_min = params["Pcrit_min"]
    R_AW = params["R_AW"]
    R_CW = params["R_CW"]
    R_L = params["R_L"]
    R_trachea = params["R_trachea"]

    a1, a2, tau, t1, t2 = updates["Nd"][-5:]

    dt = 0.001
    n_steps = int(np.round((t1 + t2) / dt)) + 1
    times = np.linspace(0, (t1 + t2), n_steps)

    V_interp = interp1d(times, updates["V_current"], kind="linear", fill_value="extrapolate")
    dVdt_interp = interp1d(times, updates["dV_dt_current"], kind="linear", fill_value="extrapolate")
    P_musc_interp = interp1d(times, updates["P_musc_current"], kind="linear", fill_value="extrapolate")

    last_breath_time = t - updates["finish_breath_time"][-1]

    # Later, for any time t:
    breath = last_breath_time % (t1 + t2)

    V = V_interp(breath)
    dV_dt = dVdt_interp(breath)
    P_musc = P_musc_interp(breath)

    if dV_dt < 0:
        P_CW = E_CW * V - 1
        P_a_dash = P_ao
    else:
        P_CW = E_CW * V - 1 + R_CW * dV_dt
        P_a_dash = P_ao - k_aw1 * dV_dt - k_aw2 * (np.abs(dV_dt)) ** 2

    if P_a_dash < 0:
        P_a = 0
    else:
        P_a = P_a_dash

    P_pl = P_CW + P_a - P_musc

    Vflow_LA = Vflow_ua + dV_dt
    P_ua = P_pl + Vflow_LA * R_rs

    if t == 0:
        dP_ua_dt = 0
    else:
        dP_ua_dt = (P_ua - updates["P_ua"][i - 1]) / (t - updates["time_history"][i - 1])

    dVflow_ua_dt = -(1 / R_trachea) * (dP_ua_dt + (1 / C_ua) * Vflow_ua)


    if num_removed > 0:
        keys = [
            "Vflow_ua", "P_ua", "P_musc", "dV_dt", "V", "P_pl"
        ]
        for key in keys:
            updates[key][(i - num_removed): (i + 1)] = np.full((num_removed + 1,), 1e6)

        i = i - num_removed


    updates["Vflow_ua"][i] = Vflow_ua
    updates["P_ua"][i] = P_ua
    updates["P_musc"][i] = P_musc
    updates["dV_dt"][i] = dV_dt
    updates["V"][i] = V
    updates["P_pl"][i] = P_pl
    updates["breath"][i] = breath

    return [dVflow_ua_dt]