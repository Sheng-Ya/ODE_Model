import bisect

import numpy as np

def gas_exchange(t, state, params, time_history, resp_mech_inputs, resp_control_inputs, heart_system_inputs, updates, all_time, num_removed):
    """
        # Gas Exchange and Mixing need inputs: Q_pp, Q_bp, Q_la, time_history, V, dV_dt

    """

    (Pd_1_O2, Pd_1_CO2, Pd_2_O2, Pd_2_CO2, Pd_3_O2, Pd_3_CO2, Pd_4_O2, Pd_4_CO2, Pd_5_O2, Pd_5_CO2, Pa_O2, Pa_CO2,
     dPa_O2_dt, dPa_CO2_dt, PA_O2, PA_CO2, PvbCO2, PCSFCO2, MRTO2, MRTCO2, CvO2, CvCO2, MRV) = state

    # Gas Exchange and Mixing
    a1 = params["a1"]
    a2 = params["a2"]
    alpha1 = params["alpha1"]
    alpha2 = params["alpha2"]
    beta1 = params["beta1"]
    beta2 = params["beta2"]
    C1 = params["C1"]
    C2 = params["C2"]
    Pd_CO2_IC = params["Pd_CO2_IC"]
    Pd_O2_IC = params["Pd_O2_IC"]
    Fi_CO2 = params["Fi_CO2"]
    Fi_O2 = params["Fi_O2"]
    K1 = params["K1"]
    K2 = params["K2"]
    LCTV = params["LCTV"]
    PACO2_Delay_IC = params["PACO2_Delay_IC"]
    dPa_CO2_dt_IC = params["dPa_CO2_dt_IC"]
    PACO2_IC = params["PACO2_IC"]
    d2Pa_CO2_dt2_IC = params["d2Pa_CO2_dt2_IC"]
    PAO2_Delay_IC = params["PAO2_Delay_IC"]
    dPa_O2_dt_IC = params["dPa_O2_dt_IC"]
    PAO2_IC = params["PAO2_IC"]
    d2Pa_O2_dt2_IC = params["d2Pa_O2_dt2_IC"]
    P_atm = params["P_atm"]
    P_ws = params["P_ws"]
    T1 = params["T1"]
    T2 = params["T2"]
    VL_CO2 = params["VL_CO2"]
    VL_O2 = params["VL_O2"]
    Z = params["Z"]

    V_dead = resp_control_inputs["VD"][-1]

    # other inputs
    dV_dt = resp_mech_inputs["dV_dt"][-1]
    V = resp_mech_inputs["V"][-1]
    Q_pp = heart_system_inputs["Q_pp"][-2]/1000
    Q_bp = heart_system_inputs["Q_bp"][-2]/1000
    Q_la = heart_system_inputs["Q_la"][-2]/1000

    for i in range(1, 6):
        if i == 1 and dV_dt >= 0:
            PiO2 = Fi_O2 * (P_atm - P_ws) / 100
            PiCO2 = Fi_CO2 * (P_atm - P_ws) / 100
            dPd_1_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (PiO2 - Pd_1_O2)
            dPd_1_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (PiCO2 - Pd_1_CO2)
        if i > 1 and dV_dt >= 0:
            dPd_2_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_1_O2 - Pd_2_O2)
            dPd_2_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_1_CO2 - Pd_2_CO2)

            dPd_3_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_2_O2 - Pd_3_O2)
            dPd_3_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_2_CO2 - Pd_3_CO2)

            dPd_4_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_3_O2 - Pd_4_O2)
            dPd_4_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_3_CO2 - Pd_4_CO2)

            dPd_5_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_4_O2 - Pd_5_O2)
            dPd_5_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_4_CO2 - Pd_5_CO2)
        if i < 5 and dV_dt < 0:
            dPd_1_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_2_O2 - Pd_1_O2)
            dPd_1_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_2_CO2 - Pd_1_CO2)

            dPd_2_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_3_O2 - Pd_2_O2)
            dPd_2_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_3_CO2 - Pd_2_CO2)

            dPd_3_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_4_O2 - Pd_3_O2)
            dPd_3_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_4_CO2 - Pd_3_CO2)

            dPd_4_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_5_O2 - Pd_4_O2)
            dPd_4_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_5_CO2 - Pd_4_CO2)
        if i == 5 and dV_dt < 0:
            dPd_5_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (PA_O2 - Pd_5_O2)
            dPd_5_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (PA_CO2 - Pd_5_CO2)


    Ta = LCTV / Q_la
    t_minus_Ta = t - Ta
    if t_minus_Ta >= 0 and time_history.size != 0:
        # Find the index for delay_time in time_history
        # index = max([i for i, t in enumerate(time_history) if t <= t_minus_Ta])
        index = bisect.bisect_right(time_history, t_minus_Ta) - 1
        PA_O2_old = updates["PA_O2"][index]
        PA_CO2_old = updates["PA_CO2"][index]
    else:
        PA_O2_old = PAO2_Delay_IC
        PA_CO2_old = PACO2_Delay_IC

    x1 = Pa_O2
    x2 = Pa_CO2
    dx1_dt = dPa_O2_dt
    dx2_dt = dPa_CO2_dt
    d2Pa_O2_dt2 = (PA_O2_old - (T1 + T2) * dx1_dt - x1) / (T1 * T2)
    d2Pa_CO2_dt2 = (PA_CO2_old - (T1 + T2) * dx2_dt - x2) / (T1 * T2)

    FCO2 = (PA_CO2 * (1 + beta2 * PA_O2)) / (K2 * (1 + alpha2 * PA_O2))
    CaCO2 = (C2 * Z) * (FCO2 ** (1/a2)) / (1 + (FCO2 ** (1/a2)))

    FO2 = (PA_O2 * (1 + beta1 * PA_CO2)) / (K1 * (1 + alpha1 * PA_CO2))
    CaO2 = (C1 * Z) * (FO2 ** (1 / a1)) / (1 + (FO2 ** (1 / a1)))


    V_O2 = V + VL_O2
    V_CO2 = V + VL_CO2

    if dV_dt >= 0:
        dPA_O2_dt = (863 * Q_pp * (CvO2 - CaO2) + dV_dt * (Pd_5_O2 - PA_O2)) / V_O2
        dPA_CO2_dt = (863 * Q_pp * (CvCO2 - CaCO2) + dV_dt * (Pd_5_CO2 - PA_CO2)) / V_CO2
    else:
        dPA_O2_dt = (863 * Q_pp * (CvO2 - CaO2)) / V_O2
        dPA_CO2_dt = (863 * Q_pp * (CvCO2 - CaCO2)) / V_CO2

    # Gas transport
    # Brain
    dc = params["dc"]
    h = params["h"]
    KCCO2 = params["KCCO2"]
    KCSFCO2 = params["KCSFCO2"]
    MRBCO2 = params["MRBCO2"]
    MRBO2 = params["MRBO2"]
    PbCO2IC = params["PbCO2IC"]
    SbCO2 = params["SbCO2"]
    SCO2 = params["SCO2"]



    Pb_CO2 = PvbCO2 + (PCSFCO2 - PvbCO2) * np.exp(-dc * ((Q_bp * KCCO2) ** 0.5))

    dPvbCO2_dt = (MRBCO2 + Q_bp * SCO2 * (Pa_CO2 - PvbCO2) - h) / SbCO2
    dPCSFCO2_dt = (PvbCO2 - PCSFCO2) / KCSFCO2

    # Body Tissues Compartment
    Cv_CO2_IC = params["Cv_CO2_IC"]
    Cv_O2_IC = params["Cv_O2_IC"]
    MRCO2 = params["MRCO2"]
    MRO2 = params["MRO2"]
    MRTCO2_basal = params["MRTCO2_basal"]
    MRTO2_basal = params["MRTO2_basal"]
    tauMR = params["tauMR"]
    VTCO2 = params["VTCO2"]
    VTO2 = params["VTO2"]

    tau_MRV = params["tau_MRV"]

    QT = Q_pp - Q_bp

    dMRTO2_dt = (MRO2 - MRTO2) / tauMR
    dMRTCO2_dt = (MRCO2 - MRTCO2) / tauMR

    dCvO2_dt = (-MRTO2 + QT * (CaO2 - CvO2)) / VTO2
    dCvCO2_dt = (MRTCO2 + QT * (CaCO2 - CvCO2)) / VTCO2

    cO2_diff = QT * (CaO2 - CvO2)
    cCO2_diff = QT * (CaCO2 - CvCO2)

    # Metabolism Dynamic
    MRR = (MRBCO2 + MRBO2 + MRTCO2 + MRTO2) / (MRBCO2 + MRBO2 + MRTCO2_basal + MRTO2_basal)

    if MRR < 1:
        MRR = 1

    if MRV < 0 or MRR <= 1:
        MRV = 0

    dMRV_dt = ((MRR - 1) - MRV) / tau_MRV

    if t != 0:
        if t < all_time[-1]:
            for key in [
                "Pb_CO2", "Pa_O2", "Pa_CO2", "MRV", "MRTCO2", "Pb_CO2_history",
                "Pa_O2_history", "Pa_CO2_history", "Ca_O2", "PA_O2", "PA_CO2", "Cv_O2", "Ca_CO2", "Cv_CO2", "FCO2", "FO2", "QT",
                "cCO2_diff", "cO2_diff", "dCvCO2_dt", "dCvO2_dt"
            ]:
                updates[key] = updates[key][:-num_removed]

    # t_eval = updates["t_eval3"][0]
    # tolerance = 1e-3
    # if np.abs(t - t_eval) < tolerance:
    updates["Pb_CO2"] = np.append(updates["Pb_CO2"], Pb_CO2)
    updates["Pa_O2"] = np.append(updates["Pa_O2"], Pa_O2)
    updates["Pa_CO2"] = np.append(updates["Pa_CO2"], Pa_CO2)
    updates["MRV"] = np.append(updates["MRV"], MRV)
    updates["MRTCO2"] = np.append(updates["MRTCO2"], MRTCO2)
    updates["Pb_CO2_history"] = np.append(updates["Pb_CO2_history"], Pb_CO2)
    updates["Pa_O2_history"] = np.append(updates["Pa_O2_history"], Pa_O2)
    updates["Pa_CO2_history"] = np.append(updates["Pa_CO2_history"], Pa_CO2)
    updates["Ca_O2"] = np.append(updates["Ca_O2"], CaO2)
    updates["Cv_O2"] = np.append(updates["Cv_O2"], CvO2)
    updates["Ca_CO2"] = np.append(updates["Ca_CO2"], CaCO2)
    updates["Cv_CO2"] = np.append(updates["Cv_CO2"], CvCO2)
    updates["PA_O2"] = np.append(updates["PA_O2"], PA_O2)
    updates["PA_CO2"] = np.append(updates["PA_CO2"], PA_CO2)
    updates["FCO2"] = np.append(updates["FCO2"], FCO2)
    updates["FO2"] = np.append(updates["FO2"], FO2)
    updates["QT"] = np.append(updates["QT"], QT)
    updates["cCO2_diff"] = np.append(updates["cCO2_diff"], cCO2_diff)
    updates["cO2_diff"] = np.append(updates["cO2_diff"], cO2_diff)
    updates["dCvO2_dt"] = np.append(updates["dCvO2_dt"], dCvO2_dt)
    updates["dCvCO2_dt"] = np.append(updates["dCvCO2_dt"], dCvCO2_dt)

    # updates["t_eval3"] = updates["t_eval3"][1:]

    return [dPd_1_O2_dt, dPd_1_CO2_dt, dPd_2_O2_dt, dPd_2_CO2_dt, dPd_3_O2_dt, dPd_3_CO2_dt, dPd_4_O2_dt,
            dPd_4_CO2_dt, dPd_5_O2_dt, dPd_5_CO2_dt, dx1_dt, dx2_dt, d2Pa_O2_dt2, d2Pa_CO2_dt2, dPA_O2_dt,
            dPA_CO2_dt, dPvbCO2_dt, dPCSFCO2_dt, dMRTO2_dt, dMRTCO2_dt, dCvO2_dt, dCvCO2_dt, dMRV_dt]