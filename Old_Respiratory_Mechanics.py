import os

import numpy as np
import pandas as pd


def respiratory_mechanics(t, state, params, exp_inputs, updates, all_time, num_removed):
    """
        Pulmonary Mechanics state variables: V
        Upper Airway state variables: alpha

    """

    (V, alpha) = state

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

    a0, a1, a2, tau, t1, t2 = exp_inputs["Nd"][-6:]

    E_rs = E_CW + E_L

    breath = t % (t1 + t2)

    if 0 <= breath <= t1:
        P_musc = a0 + a1 * breath + a2 * (breath ** 2)
    elif t1 < breath <= (t1 + t2):
        P_musc_t1 = a0 + a1 * t1 + a2 * (t1 ** 2)
        P_musc = P_musc_t1 * np.exp(-(breath - t1) / tau)


    # initial value for G_AW
    G_AW = exp_inputs["G_AW_guess"][-1]
    Vflow_ua = exp_inputs["Vflow_ua"][-1]
    P_ua = exp_inputs["P_ua"][-1]
    max_iterations = 20

    # Iterative calculation for G_AW
    for _ in range(max_iterations):

        # Calculate dV/dt using the current G_AW, minute ventilation = dV/dt
        dV_dt = (G_AW / R_rs) * ((P_musc - P_ao) - E_rs * V)

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

        tolerance = 1e-6

        for _ in range(max_iterations):
            # Airway pressure and flow
            Vflow_LA = Vflow_ua + dV_dt
            new_P_ua = P_pl + Vflow_LA * R_rs
            new_Vflow_ua = -(1 / R_trachea) * (new_P_ua + (1 / C_ua) * alpha)
            if abs(new_Vflow_ua - Vflow_ua) < tolerance and abs(new_P_ua - P_ua) < tolerance:
                Vflow_ua = new_Vflow_ua
                P_ua = new_P_ua
                break
            Vflow_ua = new_Vflow_ua
            P_ua = new_P_ua

        # Set based on fixed parameters
        Pcrit = Pcrit_min

        # Update G_AW
        if P_ua <= Pcrit:
            new_G_AW = 0
        elif (Pcrit < P_ua <= 0) and (1 - (P_ua / Pcrit)) >= 0:
            new_G_AW = A0_ua * (1 - (P_ua / Pcrit)) * K_ua
        elif P_ua > 0:
            new_G_AW = A0_ua * K_ua

        # Convergence check
        if abs(new_G_AW - G_AW) < tolerance:
            G_AW = new_G_AW
            break
        G_AW = new_G_AW


    # known: P_pl, R_rs, C_ua, R_trachea
    # solving for V_flow_LA, Vflow_ua, P_ua

    # Vflow_LA = (P_ua - P_pl) / R_rs
    # Vflow_ua = Vflow_LA - dV_dt
    # P_ua = (-R_trachea) * Vflow_ua - alpha / C_ua
    # Vflow_ua = (1/(1+R_trachea/R_rs)) * ((- (1/(R_rs * C_ua)) * alpha) - P_pl/R_rs - dV_dt)

    d_alpha_dt = Vflow_ua
    # R_rs = R_AW + R_L + R_CW

    if t != 0:
        if t < all_time[-1]:
            for key in [
                "G_AW_guess", "Vflow_ua", "P_ua", "G_AW",
                "P_musc", "dV_dt", "V", "previous_dV_dt", "P_pl"
            ]:
                del updates[key][-num_removed:]

    # data_to_append = {
    #     "P_ua": P_ua,
    #     "G_AW": G_AW, "Vflow_ua": Vflow_ua,
    #     "P_musc": P_musc, "dV_dt": dV_dt, "V": V, "P_pl": P_pl
    #
    # }
    #
    # # Define the CSV file path
    # csv_file = "output_old.csv"
    #
    # # Write headers only if the file doesn't exist
    # write_header = not os.path.exists(csv_file)
    # df = pd.DataFrame([data_to_append])
    # df.to_csv(csv_file, mode='a', index=False, header=write_header)
    # # Ensure headers are written only once
    # write_header = False


    exp_inputs["G_AW_guess"].append(G_AW)

    updates["G_AW"].append(G_AW)
    updates["Vflow_ua"].append(Vflow_ua)
    updates["P_ua"].append(P_ua)
    updates["P_musc"].append(P_musc)
    updates["dV_dt"].append(dV_dt)
    updates["V"].append(V)
    updates["previous_dV_dt"].append(dV_dt)
    updates["P_pl"].append(P_pl)

    return [dV_dt, d_alpha_dt]