def gas_exchange1(t, state, params, time_history, resp_mech_inputs, resp_control_inputs, heart_system_inputs):
    """
        # Gas Exchange and Mixing need inputs: Q_pp, Q_bp, Q_la, time_history, V, dV_dt

    """

    (Pd_1_O2) = state

    # Gas Exchange and Mixing
    Fi_O2 = params["Fi_O2"]
    P_atm = params["P_atm"]
    P_ws = params["P_ws"]


    V_dead = resp_control_inputs["VD"]

    # other inputs
    dV_dt = resp_control_inputs["dV_dt"]


    for i in range(1, 6):
        if i == 1 and dV_dt >= 0:
            P1O2 = Fi_O2 * (P_atm - P_ws) / 100
            dPd_1_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (P1O2 - Pd_1_O2)


    return [dPd_1_O2_dt]