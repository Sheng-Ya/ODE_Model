import bisect
import numpy as np
# from Selected_Conditions import Selected_Conditions as previous_Selected_Conditions


def gas_exchange(t, state, params, time_history, resp_control_inputs, heart_system_inputs, updates, num_removed, i, t_start, previous_Selected_Conditions):
    """
        # Gas Exchange and Mixing need inputs: Q_pp, Q_bp, Q_la, time_history, V, dV_dt

    """

    (Pd_1_O2, Pd_1_CO2, Pd_2_O2, Pd_2_CO2, Pd_3_O2, Pd_3_CO2, Pd_4_O2, Pd_4_CO2, Pd_5_O2, Pd_5_CO2, Pa_O2, Pa_CO2,
     dPa_O2_dt, dPa_CO2_dt, PA_O2, PA_CO2, PCSFCO2, MRTO2, MRTCO2, CTO2, CvtCO2, CBO2, CvbCO2, MRV) = state

    # Gas Exchange and Mixing
    a2 = params["a2"]
    alpha1 = params["alpha1"]
    alpha2 = params["alpha2"]
    beta1 = params["beta1"]
    beta2 = params["beta2"]
    C2 = params["C2"]
    Fi_CO2 = params["Fi_CO2"]
    Fi_O2 = params["Fi_O2"]
    K1 = params["K1"]
    K2 = params["K2"]
    LCTV = params["LCTV"]
    PACO2_Delay_IC = params["PACO2_Delay_IC"]
    PAO2_Delay_IC = params["PAO2_Delay_IC"]
    P_atm = params["P_atm"]
    P_ws = params["P_ws"]
    T1 = params["T1"]
    T2 = params["T2"]
    VL_CO2 = params["VL_CO2"]
    VL_O2 = params["VL_O2"]
    Z = params["Z"]
    s = 0.04

    if t == t_start:
        heart_index = i
        resp_mech_index = i
        resp_control_index = i
    # elif num_removed > 0:
    #     heart_index = i - num_removed - 1
    #     # resp_mech variables have not been removed yet
    #     resp_mech_index = i - 1
    #     resp_control_index = i - 1
    else:
        heart_index = i - 1 - num_removed
        resp_mech_index = i - 1 - num_removed
        resp_control_index = i - 1 - num_removed

    # inputs
    V_dead = resp_control_inputs["VD"][resp_control_index] # need to change once resp controller is added in
    dV_dt = resp_control_inputs["dV_dt"][resp_control_index]
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

            dPd_5_O2_dt = (abs(dV_dt) / (0.2 * V_dead)) * (Pd_4_O2 - Pd_5_O2) # edited to just have one deadspace
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
    Ta = 2

    t_minus_Ta = t - Ta

    if t_minus_Ta >= t_start and t > abs(Ta) and Ta > 0:
        # Find the index for delay_time in time_history
        delay_index = bisect.bisect_right(time_history, t_minus_Ta) - 1
        PA_O2_old = updates["PA_O2"][delay_index]
        PA_CO2_old = updates["PA_CO2"][delay_index]
    else:
        if t == 0:
            PA_O2_old = PAO2_Delay_IC
            PA_CO2_old = PACO2_Delay_IC
        else:
            if t_start != 0 and t > abs(Ta):
                if t == t_start:
                    PA_O2_old = previous_Selected_Conditions["PA_O2"][-1]
                    PA_CO2_old = previous_Selected_Conditions["PA_CO2"][-1]
                else:
                    delay_index = bisect.bisect_right(previous_Selected_Conditions["time_history"], t_minus_Ta) - 1
                    PA_O2_old = previous_Selected_Conditions["PA_O2"][delay_index]
                    PA_CO2_old = previous_Selected_Conditions["PA_CO2"][delay_index]
            else:
                if t == t_start:
                    PA_O2_old = previous_Selected_Conditions["PA_O2"][-1]
                    PA_CO2_old = previous_Selected_Conditions["PA_CO2"][-1]
                else:
                    PA_O2_old = updates["PA_O2_old"][i - 1 - num_removed]
                    PA_CO2_old = updates["PA_CO2_old"][i - 1 - num_removed]

    x1 = Pa_O2
    x2 = Pa_CO2
    dx1_dt = dPa_O2_dt
    dx2_dt = dPa_CO2_dt
    d2Pa_O2_dt2 = (PA_O2_old - (T1 + T2) * dx1_dt - x1) / (T1 * T2)
    d2Pa_CO2_dt2 = (PA_CO2_old - (T1 + T2) * dx2_dt - x2) / (T1 * T2)

    FCO2 = (PA_CO2 * (1 + beta2 * PA_O2)) / (K2 * (1 + alpha2 * PA_O2))
    CeCO2 = (C2 * Z) * (FCO2 ** (1 / a2)) / (1 + (FCO2 ** (1 / a2)))

    alpha_O2 = 0.0000317
    alpha_CO2 = 0.000667

    FO2 = (PA_O2 * (1 + beta1 * PA_CO2)) / (K1 * (1 + alpha1 * PA_CO2))
    # CaO2_1 = (C1 * Z) * (FO2 ** (1 / a1)) / (1 + (FO2 ** (1 / a1)))
    PAO2_virt = PA_O2 * (40/PA_CO2) ** 0.3
    SaO2 = (PAO2_virt**2.6)/(PAO2_virt**2.6 + 26.6**2.6)
    CeO2 = (0.00134 * 150 * SaO2)  + 3.03e-5 * PA_O2
    # CaO2 = CaO2 * 0.8

    # BB = 46.2  + 0.31 * 0.6206 * 150 * (1 - SaO2)# mmol/L
    # Pr_tot = 38.9 # mmol/L
    # gamma = 15.84
    # CaCO2_mmol = ((BB - Pr_tot) / 2 + ((1 - gamma / 2) * alpha_CO2 * PA_CO2) / Z +
    #          0.5 * ((BB - Pr_tot) ** 2 + 2 * (BB + Pr_tot) * (gamma * alpha_CO2 * PA_CO2) / Z + (gamma * alpha_CO2 / Z * PA_CO2) ** 2) ** 0.5)
    # CaCO2 = CaCO2_mmol * Z

    # Gas transport
    # Brain
    dc = params["dc"]
    KCCO2 = params["KCCO2"]
    KCSFCO2 = params["KCSFCO2"]
    MRBCO2 = params["MRBCO2"]
    MRBO2 = params["MO2_bp"]/1000
    VB = params["VB"]

    # Body Tissues Compartment
    MRTCO2_basal = params["MRTCO2_basal"] - params["MRBCO2"]
    MRTO2_basal = params["MRTO2_basal"] - params["MO2_bp"]/1000
    tauMR = params["tauMR"]
    VTCO2 = params["VTCO2"]
    VTO2 = params["VTO2"]

    # exercise
    MRCO2 = params["MRCO2"] - params["MRBCO2"]
    MRO2 = params["MRO2"] - params["MO2_bp"]/1000

    if 1000 < t <= 1500:
        MRCO2 = 0.4/60 - 0.0009
        MRO2 = 0.45 / 60 - 0.000925

    # if 1200 < t <= 1350:
    #     MRCO2 = 0.6/60 - 0.0009
    #     MRO2 = 0.65 / 60 - 0.000925
    #
    # if 1350 < t <= 1500:
    #     MRCO2 = 0.8/60 - 0.0009
    #     MRO2 = 0.85 / 60 - 0.000925
    #
    # if 910 < t:
    #     MRCO2 = 1/60 - 0.0009
    #     MRO2 = 1.05 / 60 - 0.000925

    ## new code
    # PvbCO2 and PvbO2 is the same as the brain compartment CO2 and O2 partial pressure
    # CvbO2 is NOT the same as CBO2 (CBO2 doesn't include haemoglobin), but here CvbCO2 is the SAME as CBCO2 (just the curve)

    # brain
    PvbO2 = CBO2 / alpha_O2  # henry
    PvbCO2 = ((CvbCO2 / (C2 * Z - CvbCO2)) ** a2) * (K2 * (1 + alpha2 * PvbO2)) / (
                1 + beta2 * PvbO2)  # haldane effect/ CO2 dissociation curve

    # FbO2 = (PvbO2 * (1 + beta1 * PvbCO2)) / (K1 * (1 + alpha1 * PvbCO2))  # bohr curve
    # CvbO2_1 = (C1 * Z) * (FbO2 ** (1 / a1)) / (1 + (FbO2 ** (1 / a1)))  # bohr curve

    PvbO2_virt = PvbO2 * (40/PvbCO2) ** 0.3
    SvbO2 = (PvbO2_virt ** 2.6) / (PvbO2_virt ** 2.6 + 26.6 ** 2.6)
    CvbO2 = 0.00134 * 150 * SvbO2 + 3.03e-5 * PvbO2

    # tissue
    PvtO2 = CTO2 / alpha_O2  # henry
    PvtCO2 = ((CvtCO2 / (C2 * Z - CvtCO2)) ** a2) * (K2 * (1 + alpha2 * PvtO2)) / (
                1 + beta2 * PvtO2)  # haldane effect/ CO2 dissociation curve

    # serna and carlos
    # FtO2 = (PvtO2 * (1 + beta1 * PvtCO2)) / (K1 * (1 + alpha1 * PvtCO2))  # bohr curve
    # CvtO2_1 = (C1 * Z) * (FtO2 ** (1 / a1)) / (1 + (FtO2 ** (1 / a1)))  # bohr curve
    # ursino model 1997
    PvtO2_virt = PvtO2 * (40/PvtCO2) ** 0.3
    SvtO2 = (PvtO2_virt ** 2.6) / (PvtO2_virt ** 2.6 + 26.6 ** 2.6)
    CvtO2 = 0.00134 * 150 * SvtO2 + 3.03e-5 * PvtO2

    QT = Q_pp - Q_bp

    # overall CvO2 and CvCO2
    CvO2 = (Q_bp / Q_pp) * CvbO2 + (QT / Q_pp) * CvtO2
    CvCO2 = (Q_bp / Q_pp) * CvbCO2 + (QT / Q_pp) * CvtCO2

    CaO2 = (1 - s) * CeO2 + s * CvO2
    CaCO2 = (1 - s) * CeCO2 + s * CvCO2

    dCBO2_dt = (-MRBO2 + Q_bp * (CaO2 - CvbO2)) / VB  # brain volume for conc is 0.9
    dCvbCO2_dt = (MRBCO2 + Q_bp * (CaCO2 - CvbCO2)) / VB  # brain volume for conc is 0.9

    dCTO2_dt = (-MRTO2 + QT * (CaO2 - CvtO2)) / VTO2
    dCvtCO2_dt = (MRTCO2 + QT * (CaCO2 - CvtCO2)) / VTCO2

    Pb_CO2 = PvbCO2 + (PCSFCO2 - PvbCO2) * np.exp(-dc * ((Q_bp * KCCO2) ** 0.5))
    # Pb_CO2 = 43
    # dPvbCO2_dt = (MRBCO2 + Q_bp * SCO2 * (Pa_CO2 - PvbCO2) - h) / SbCO2
    dPCSFCO2_dt = (PvbCO2 - PCSFCO2) / KCSFCO2

    tau_MRV = params["tau_MRV"]
    dMRTO2_dt = (MRO2 - MRTO2) / tauMR
    dMRTCO2_dt = (MRCO2 - MRTCO2) / tauMR

    cO2_diff = QT * (CaO2 - CvtO2)
    cCO2_diff = QT * (CaCO2 - CvtCO2)

    V_O2 = VL_O2 # removed + V as this helps decrease VAflow (decreased time constant for ventilation)
    V_CO2 = VL_CO2

    if dV_dt >= 0:  # deadspace PAO2 is increasing towards 150
        dPA_O2_dt = (863 * Q_pp * (CvO2 - CaO2) * (1 - s) + dV_dt * (Pd_5_O2 - PA_O2)) / V_O2 # 863 is unit conversion. First from stpd to btps (x 1.21), then into pressure (x 713, P_atm - P_h20)
        dPA_CO2_dt = (863 * Q_pp * (CvCO2 - CaCO2) * (1 - s) + dV_dt * (Pd_5_CO2 - PA_CO2)) / V_CO2
        # dPA_O2_dt = -dPA_CO2_dt
    else:  # deadspace PAO2 is decreasing towards PA_O2 during expiration
        dPA_O2_dt = (863 * Q_pp * (CvO2 - CaO2) * (1 - s)) / V_O2
        dPA_CO2_dt = (863 * Q_pp * (CvCO2 - CaCO2) * (1 - s)) / V_CO2
        # dPA_O2_dt = -dPA_CO2_dt

    # Metabolism Dynamic
    MRR = (MRBCO2 + MRBO2 + MRTCO2 + MRTO2) / (MRBCO2 + MRBO2 + MRTCO2_basal + MRTO2_basal)

    if MRR < 1:
        MRR = 1

    if MRV < 0 or MRR <= 1:
        MRV = 0

    dMRV_dt = ((MRR - 1) - MRV) / tau_MRV

    if num_removed > 0:
        keys = [
            "MRTCO2", "Pa_O2", "Pa_CO2", "Ca_O2", "MRV", "PA_O2", "PA_CO2", "PA_O2_old", "PA_CO2_old",
            # "Pb_CO2", "Cv_O2", "Ca_CO2", "Cv_CO2", "FCO2", "FO2", "QT",
            # "cCO2_diff", "cO2_diff", "Ta", "dPA_CO2_dt", "dPA_O2_dt", "Pd_5_O2", "Pd_5_CO2",
            # "t_minus_Ta", "PvtCO2"
        ]
        keys2 = [
            "Pb_CO2_history", "Pa_O2_history", "Pa_CO2_history"
        ]
        for key in keys:
            updates[key][(i - num_removed): (i + 1)] = np.full((num_removed + 1,), 1e6)
        for key in keys2:
            del updates[key][-num_removed:]

        i = i - num_removed

    # cardio control inputs
    updates["MRTCO2"][i] = MRTCO2
    updates["Pa_O2"][i] = Pa_O2
    updates["Pa_CO2"][i] = Pa_CO2
    updates["Ca_O2"][i] = CaO2

    # resp control vent inputs
    updates["MRV"][i] = MRV

    # histories for gas
    updates["PA_O2"][i] = PA_O2
    updates["PA_CO2"][i] = PA_CO2
    updates["PA_O2_old"][i] = PA_O2_old
    updates["PA_CO2_old"][i] = PA_CO2_old

    if t == 0:
        updates["Pa_O2_history"].clear()
        updates["Pa_CO2_history"].clear()
        updates["Pb_CO2_history"].clear()


    updates["Pb_CO2_history"].append((t, Pb_CO2))
    updates["Pa_O2_history"].append((t, Pa_O2))
    updates["Pa_CO2_history"].append((t, Pa_CO2))

    # A = updates["Pa_CO2_history"]


    # just for plotting purposes
    updates["Pd_5_O2"][i] = Pd_5_O2
    updates["Pb_CO2"][i] = Pb_CO2
    updates["Cv_O2"][i] = CvO2
    updates["Ca_CO2"][i] = CaCO2
    updates["Cv_CO2"][i] = CvCO2
    updates["FCO2"][i] = FCO2
    updates["FO2"][i] = FO2
    updates["QT"][i] = QT
    updates["cCO2_diff"][i] = cCO2_diff
    updates["cO2_diff"][i] = cO2_diff
    updates["Ta"][i] = Ta
    updates["dPA_O2_dt"][i] = dPA_O2_dt
    updates["dPA_CO2_dt"][i] = dPA_CO2_dt
    updates["Pd_5_O2"][i] = Pd_5_O2
    updates["Pd_5_CO2"][i] = Pd_5_CO2
    updates["t_minus_Ta"][i] = t_minus_Ta
    updates["PvtCO2"][i] = PvtCO2

    return [dPd_1_O2_dt, dPd_1_CO2_dt, dPd_2_O2_dt, dPd_2_CO2_dt, dPd_3_O2_dt, dPd_3_CO2_dt, dPd_4_O2_dt,
            dPd_4_CO2_dt, dPd_5_O2_dt, dPd_5_CO2_dt, dx1_dt, dx2_dt, d2Pa_O2_dt2, d2Pa_CO2_dt2, dPA_O2_dt,
            dPA_CO2_dt, dPCSFCO2_dt, dMRTO2_dt, dMRTCO2_dt, dCTO2_dt, dCvtCO2_dt, dCBO2_dt, dCvbCO2_dt, dMRV_dt]