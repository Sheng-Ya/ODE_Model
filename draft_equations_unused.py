import numpy as np

A = np.load("IC_final.npy", allow_pickle=False)

state_names = [
    # Cardio state variables
    "VT_pa", "VT_pp", "VT_pv", "Q_pa",
    "VT_la", "VT_lv", "VT_ra", "VT_rv",
    "VT_sv", "VT_bv", "VT_hv", "VT_rmv", "VT_amv", "P_sp", "P_sa", "Q_sa", "VT_vc",
    "theta_ao", "dtheta_ao_dt", "theta_po", "dtheta_po_dt", "theta_mi", "dtheta_mi_dt",
    "theta_tr", "dtheta_tr_dt",

    # Cardio controller
    "theta_change_O2_sp", "theta_change_CO2_sp", "theta_change_O2_sv", "theta_change_CO2_sv", "theta_change_O2_sh",
    "theta_change_CO2_sh", "P_tilda", "f_ac", "f_ap", "R_ep_change", "R_sp_change",
    "R_rmp_n_change", "R_amp_n_change", "Vu_ev_change", "Vu_sv_change", "Vu_rmv_change", "Vu_amv_change",
    "Emax_lv_change", "Emax_rv_change", "Ts_change", "Tv_change", "xb_O2", "xb_CO2", "xh_O2", "xh_CO2", "Wh",
    "xrm_O2", "xrm_CO2", "xam_O2", "xM", "x_met", "P_n_current",

    # Gas exchange
    "Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2", "Pd_4_O2", "Pd_4_CO2", "Pd_5_O2",
    "Pd_5_CO2", "Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2", "PCSFCO2", "MRTO2", "MRTCO2",
    "CTO2", "CvtCO2", "CBO2", "CvbCO2", "MRV",

    # Respiratory control
    "VE_integral"
]

Initial_Conditions = {name: float(value) for name, value in zip(state_names, A)}
Initial_Conditions = dict(sorted(Initial_Conditions.items(), key=lambda x: x[0]))


print(Initial_Conditions)