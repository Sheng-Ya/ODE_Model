import bisect
import os

import numpy as np
import pandas as pd


def gas_exchange(t, state, params, time_history, resp_mech_inputs, resp_control_inputs, heart_system_inputs, updates, num_removed, i):
    """
        # Gas Exchange and Mixing need inputs: Q_pp, Q_bp, Q_la, time_history, V, dV_dt

    """

    (Pd_1_O2, Pd_1_CO2, Pd_2_O2, Pd_2_CO2, Pd_3_O2, Pd_3_CO2, Pd_4_O2, Pd_4_CO2, Pd_5_O2, Pd_5_CO2, Pa_O2, Pa_CO2,
     dPa_O2_dt, dPa_CO2_dt, PA_O2, PA_CO2, PvbCO2, PCSFCO2, MRTO2, MRTCO2, CvO2, CvCO2, MRV, CbCO2, CbO2) = state

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

    if t == 0:
        heart_index = i
        resp_mech_index = i
        # resp_control_index = 0
        resp_control_index = i
    elif num_removed > 0:
        heart_index = i - num_removed - 1
        # resp_mech variables have not been removed yet
        resp_mech_index = i - 1
        # resp_control_index = 0
        resp_control_index = i - 1
    else:
        heart_index = i - 1
        resp_mech_index = i - 1
        # resp_control_index = 0
        resp_control_index = i - 1

    V_dead = resp_control_inputs["VD"][resp_control_index] # need to change once resp controller is added in

    # other inputs
    dV_dt = resp_mech_inputs["dV_dt"][resp_mech_index]
    V = resp_mech_inputs["V"][resp_mech_index]
    Q_pp = heart_system_inputs["Q_pp"][heart_index]/1000
    Q_bp = heart_system_inputs["Q_bp"][heart_index]/1000
    Q_la = heart_system_inputs["Q_la"][heart_index]/1000

    for w in range(1, 6):
        if w == 1 and dV_dt >= 0:
            PiO2 = Fi_O2 * (P_atm - P_ws) / 100
            PiCO2 = Fi_CO2 * (P_atm - P_ws) / 100
            dPd_1_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (PiO2 - Pd_1_O2)
            dPd_1_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (PiCO2 - Pd_1_CO2)
        if w > 1 and dV_dt >= 0:
            dPd_2_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_1_O2 - Pd_2_O2)
            dPd_2_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_1_CO2 - Pd_2_CO2)

            dPd_3_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_2_O2 - Pd_3_O2)
            dPd_3_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_2_CO2 - Pd_3_CO2)

            dPd_4_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_3_O2 - Pd_4_O2)
            dPd_4_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_3_CO2 - Pd_4_CO2)

            dPd_5_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_4_O2 - Pd_5_O2)
            dPd_5_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_4_CO2 - Pd_5_CO2)
        if w < 5 and dV_dt < 0:
            dPd_1_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_2_O2 - Pd_1_O2)
            dPd_1_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_2_CO2 - Pd_1_CO2)

            dPd_2_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_3_O2 - Pd_2_O2)
            dPd_2_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_3_CO2 - Pd_2_CO2)

            dPd_3_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_4_O2 - Pd_3_O2)
            dPd_3_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_4_CO2 - Pd_3_CO2)

            dPd_4_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_5_O2 - Pd_4_O2)
            dPd_4_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_5_CO2 - Pd_4_CO2)
        if w == 5 and dV_dt < 0:
            dPd_5_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (PA_O2 - Pd_5_O2)
            dPd_5_CO2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (PA_CO2 - Pd_5_CO2)


    Ta = LCTV / Q_la

    if t > 0:
        Ta = 0.01 * Ta + 0.99 * updates["Ta"][i - 1]

    t_minus_Ta = t - Ta

    if t_minus_Ta >= 0 and t > abs(Ta):
        # Find the index for delay_time in time_history
        delay_index = bisect.bisect_right(time_history, t_minus_Ta) - 1
        PA_O2_old = updates["PA_O2"][delay_index]
        PA_CO2_old = updates["PA_CO2"][delay_index]
        # PA_O2_old = updates["PA_O2"][i - 1]
        # PA_CO2_old = updates["PA_CO2"][i - 1]
    else:
        if t == 0:
            PA_O2_old = PAO2_Delay_IC
            PA_CO2_old = PACO2_Delay_IC
        else:
            PA_O2_old = updates["PA_O2_old"][i-1]
            PA_CO2_old = updates["PA_CO2_old"][i-1]

    x1 = Pa_O2
    x2 = Pa_CO2
    dx1_dt = dPa_O2_dt
    dx2_dt = dPa_CO2_dt
    d2Pa_O2_dt2 = (PA_O2_old - (T1 + T2) * dx1_dt - x1) / (T1 * T2)
    d2Pa_CO2_dt2 = (PA_CO2_old - (T1 + T2) * dx2_dt - x2) / (T1 * T2)

    # PA_CO2 = 40 + np.cos(t)
    # PA_O2 = 100 + np.cos(t)

    FCO2 = (PA_CO2 * (1 + beta2 * PA_O2)) / (K2 * (1 + alpha2 * PA_O2))
    # FCO2 = 1 + 0.25* np.cos(t)
    CaCO2 = (C2 * Z) * (FCO2 ** (1 / a2)) / (1 + (FCO2 ** (1 / a2)))

    FO2 = (PA_O2 * (1 + beta1 * PA_CO2)) / (K1 * (1 + alpha1 * PA_CO2))
    # FO2 = 4.5 + np.cos(t)
    CaO2 = (C1 * Z) * (FO2 ** (1 / a1)) / (1 + (FO2 ** (1 / a1)))

    MRBCO2 = params["MRBCO2"]
    MRBO2 = params["MRBO2"]
    VB = 0.2

    dCbCO2_dt = (MRBCO2 + Q_bp * (CaCO2 - CbCO2)) / VB
    dCbO2_dt = (-MRBO2 + Q_bp * (CaO2 - CbO2))/ VB

    V_O2 = V + VL_O2
    V_CO2 = V + VL_CO2

    QT = Q_pp - Q_bp

    if t > 30 and dV_dt > 0.5:
        A = 2

    if dV_dt >= 0: # deadspace PAO2 is increasing towards 150
        dPA_O2_dt = (863 * Q_pp * (CvO2 - CaO2) + dV_dt * (Pd_5_O2 - PA_O2) + 863 * Q_bp * (CbO2 - CaO2)) / V_O2 # 863 is unit conversion from btps to stpd
        dPA_CO2_dt = (863 * Q_pp * (CvCO2 - CaCO2) + dV_dt * (Pd_5_CO2 - PA_CO2) + 863 * Q_bp * (CbCO2 - CaCO2)) / V_CO2
        # dPA_O2_dt = -dPA_CO2_dt
    else: # deadspace PAO2 is decreasing towards PA_O2 during expiration
        dPA_O2_dt = (863 * Q_pp * (CvO2 - CaO2) + 863 * Q_bp * (CbO2 - CaO2)) / V_O2
        dPA_CO2_dt = (863 * Q_pp * (CvCO2 - CaCO2) + 863 * Q_bp * (CbCO2 - CaCO2)) / V_CO2
        # dPA_O2_dt = -dPA_CO2_dt

    # Gas transport
    # Brain
    dc = params["dc"]
    h = params["h"]
    KCCO2 = params["KCCO2"]
    KCSFCO2 = params["KCSFCO2"]
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

    if num_removed > 0:
        keys = [
            "Pb_CO2", "Pa_O2", "Pa_CO2", "MRV", "MRTCO2",
            "Ca_O2", "PA_O2", "PA_CO2", "Cv_O2", "Ca_CO2", "Cv_CO2", "FCO2", "FO2", "QT",
            "cCO2_diff", "cO2_diff", "dCvCO2_dt", "dCvO2_dt", "Ta", "dPA_CO2_dt", "dPA_O2_dt", "Pd_5_O2", "Pd_5_CO2",
            "t_minus_Ta", "PA_O2_old", "PA_CO2_old", "CbO2"
        ]
        for key in keys:
            updates[key][(i - num_removed): (i + 1)] = np.full((num_removed + 1,), 1e6)

        i = i - num_removed

    # data_to_append = {
    #     "Pb_CO2": Pb_CO2, "Pa_O2": Pa_O2, "Pa_CO2": Pa_CO2,
    #     "MRV": MRV, "MRTCO2": MRTCO2,
    #     "Ca_O2": CaO2, "PA_O2": PA_O2, "PA_CO2": PA_CO2,
    #     "Cv_O2": CvO2, "Ca_CO2": CaCO2, "Cv_CO2": CvCO2,
    #     "FCO2": FCO2, "FO2": FO2, "QT": QT,
    #     "cCO2_diff": cCO2_diff, "cO2_diff": cO2_diff,
    #     "dCvCO2_dt": dCvCO2_dt, "dCvO2_dt": dCvO2_dt
    # }
    #
    # # Define the CSV file path
    # csv_file = "output.csv"
    #
    # # Write headers only if the file doesn't exist
    # write_header = not os.path.exists(csv_file)
    # df = pd.DataFrame([data_to_append])
    # df.to_csv(csv_file, mode='a', index=False, header=write_header)
    # # Ensure headers are written only once
    # write_header = False


    updates["Pb_CO2"][i] = Pb_CO2
    updates["Pa_O2"][i] = Pa_O2
    updates["Pa_CO2"][i] = Pa_CO2
    updates["MRV"][i] = MRV
    updates["MRTCO2"][i] = MRTCO2
    updates["Pb_CO2_history"].append(Pb_CO2)
    updates["Pa_O2_history"].append(Pa_O2)
    updates["Pa_CO2_history"].append(Pa_CO2)
    updates["Ca_O2"][i] = CaO2
    updates["Cv_O2"][i] = CvO2
    updates["Ca_CO2"][i] = CaCO2
    updates["Cv_CO2"][i] = CvCO2
    updates["PA_O2"][i] = PA_O2
    updates["PA_CO2"][i] = PA_CO2
    updates["FCO2"][i] = FCO2
    updates["FO2"][i] = FO2
    updates["QT"][i] = QT
    updates["cCO2_diff"][i] = cCO2_diff
    updates["cO2_diff"][i] = cO2_diff
    updates["dCvO2_dt"][i] = dCvO2_dt
    updates["dCvCO2_dt"][i] = dCvCO2_dt
    updates["Ta"][i] = Ta
    updates["dPA_O2_dt"][i] = dPA_O2_dt
    updates["dPA_CO2_dt"][i] = dPA_CO2_dt
    updates["Pd_5_O2"][i] = Pd_5_O2
    updates["Pd_5_CO2"][i] = Pd_5_CO2
    updates["t_minus_Ta"][i] = t_minus_Ta
    updates["PA_O2_old"][i] = PA_O2_old
    updates["PA_CO2_old"][i] = PA_CO2_old
    updates["CbO2"][i] = CbO2


    return [dPd_1_O2_dt, dPd_1_CO2_dt, dPd_2_O2_dt, dPd_2_CO2_dt, dPd_3_O2_dt, dPd_3_CO2_dt, dPd_4_O2_dt,
            dPd_4_CO2_dt, dPd_5_O2_dt, dPd_5_CO2_dt, dx1_dt, dx2_dt, d2Pa_O2_dt2, d2Pa_CO2_dt2, dPA_O2_dt,
            dPA_CO2_dt, dPvbCO2_dt, dPCSFCO2_dt, dMRTO2_dt, dMRTCO2_dt, dCvO2_dt, dCvCO2_dt, dMRV_dt, dCbCO2_dt, dCbO2_dt]