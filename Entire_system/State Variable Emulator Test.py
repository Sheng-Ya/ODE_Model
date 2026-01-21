import joblib
import torch
import numpy as np
from autoemulate import AutoEmulate
from SALib import ProblemSpec


X_all = np.load(f'LHCS_20000_X_rest_no_Pthor.npy')
Result_all = np.load(f'LHCS_20000_Result_rest_no_Pthor.npy')
Result_states = np.load(f'States_chunked.npy')

lower = 0.5
upper = 1.5
size = 5000

#
sp = ProblemSpec({
    'names': [
        # gas
        "beta2", "C2", "K2", "a2",
        "alpha2", "dc", "KCCO2", "GV_dead",
        # resp control
        "KcCO2", "KcMRV", "KpCO2", "KpO2",
        "V0_dead", "VA_rest", "Pmax", "Pmax_dot",
        "E_rs", "R_rs",
        # cardio
        "C_jp", "C_sa", "L_sa", "R_sa",
        "C_amv", "C_bv", "C_ev", "C_hv",
        "C_rmv", "C_sv", "kr_am", "P_0",
        "R_amv_n", "R_bv_n", "R_ev_n", "R_hv_n",
        "R_rmv_n", "R_sv_n", "D1", "K1_vc",
        "Kr_vc", "Rvc_n", "C_pa", "C_pp",
        "C_pv", "L_pa", "R_pa", "R_pp",
        "R_pv", "Emax_la", "P0_la", "Emax_ra",
        "P0_ra", "KE_la", "KE_ra", "P0_lv",
        "P0_rv",  # "g_thor", "P_thormax_n", "P_thormin_n",
        "VT_n", "s",
        # cardio control
        "fab_o", "fes_o", "fes_inf", "fes_max",
        "fev_o", "fev_inf", "kes", "kev",
        "Io_sh", "Io_sp", "Io_sv", "Io_v",
        "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v",
        "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min",
        "Ysv_max", "Ysv_min", "Yv_max", "Yv_min",
        "theta_v", "Wb_sh", "Wb_sp", "Wb_sv",
        "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v",
        "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v",
        "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
        "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv",
        "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp",
        "GR_sp", "GV_amv", "GV_ev", "GV_rmv",
        "GV_sv", "R_amp0", "R_ep0", "R_rmp0",
        #
        "R_sp0", "AT", "g_ccsh", "g_ccsp",
        "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv",
        "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn",
        "theta_spn", "theta_svn", "x_sh", "x_sp",
        "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min",
        "k_ab", "P_n", "P_n_max", "f_acCO2_n",
        "f_ac_max", "f_ac_min", "k_ac", "K_H",
        "PaO2_ac_n", "G_ap", "GT_s", "GT_v",
        "T0", "A", "B", "C",
        "D", "Cvb_O2_n", "gb_O2", "MO2_bp",
        "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2",
        "grm_O2", "Kh_CO2", "Krm_CO2", "MO2_hpn",
        "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n",
        "gam_O2", "gM", "Io_met", "kmet",
        "MO2_ampn", "phi_max", "phi_min",
        # added params
        "Kp_ao", "Kf_ao", "Kb_ao", "Kv_ao", "theta_ao_max",
        "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max",
        "Kp_po", "Kf_po", "Kb_po", "Kv_po", "theta_po_max",
        "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max",
        "alpha_O2", "R_po", "R_mi", "R_tr",
        "R_ao", "C_O2_param1", "C_O2_param2", "C_O2_param3",
        "PAMO2_nominal", "Vu_sa", "Vu_bv", "Vu_hv",
        "Vu_jp", "Vu_vc", "Vvc_max", "Vu_pa",
        "Vu_pp", "Vu_pv", "Vu_la", "Vu_lv",
        "Vu_ra", "Vu_rv",

        "V_tot", "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp",
        "tau_Rep", "tau_Rrmp", "tau_Rsp", "tau_Vamv",
        "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0",
        "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc",
        "tau_isc", "tau_p", "tau_z", "tau_ac",
        "tau_ap", "tau_Ts", "tau_Tv", "tau_CO2",
        "tau_O2", "tau_w", "tau_M", "tau_met",
        "DEmax_lv", "DEmax_rv", "DR_amp", "DR_ep",
        "DR_rmp", "DR_sp", "DV_amv", "DV_ev",
        "DV_rmv", "DV_sv", "DT_s", "DT_v",
        "Dmet", "Ta", "KE_lv", "KE_rv",
        "T1", "T2", "VL_CO2", "VL_O2",
        "KCSFCO2", "VB", "tauMR", "VTCO2",
        "VTO2", "tau_MRV",

        # further added
        "scale_param1", "scale_param2", "scale_param3", "scale_param4",
        "scale_param5", "scale_param6", "scale_param7", "Pa_O2_lower",
        "rise_time_atr", "rise_time_ven", "fall_time_ven", "ahead1",
        "theta_min", "r", "l", "V_nominal", "V_scale"
    ],

    'bounds': [
        # gas
        [0.03255 * 0.9, 0.03255 * 1.1], [87 * 0.9, 87 * 1.1], [194.4 * 0.9, 194.4 * 1.1], [1.819 * 0.9, 1.819 * 1.1],
        [0.05591 * 0.9, 0.05591 * 1.1], [0.015 * lower, 0.015 * upper], [346000 * lower, 346000 * upper], [0.1698 * lower, 0.1698 * upper],
        # resp control
        [0.2332 * lower, 0.2332 * upper], [1 * lower, 1 * upper], [0.2025 * lower, 0.2025 * upper], [4.72e-09 * lower, 4.72e-09 * upper],
        [0.1587 * lower, 0.1587 * upper], [0.0673 * lower, 0.0673 * upper], [100 * lower, 100 * upper], [1000 * lower, 1000 * upper],
        [21.9 * lower, 21.9 * upper], [3.02 * lower, 3.02 * upper],
        # cardio
        [3.72 * lower, 3.72 * upper], [0.28 * lower, 0.28 * upper], [0.00022 * lower, 0.00022 * upper], [0.2 * lower, 0.2 * upper],
        [4.4 * lower, 4.4 * upper], [5.71 * lower, 5.71 * upper], [10 * lower, 10 * upper], [1.57 * lower, 1.57 * upper],
        [3.28 * lower, 3.28 * upper], [31.11 * lower, 31.11 * upper], [24.17 * lower, 24.17 * upper], [3.93 * lower, 3.93 * upper],
        [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper], [0.224 * lower, 0.224 * upper],
        [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper], [0.3855 * lower, 0.3855 * upper], [0.15 * lower, 0.15 * upper],
        [0.0001 * lower, 0.0001 * upper], [0.0025 * lower, 0.0025 * upper], [0.76 * lower, 0.76 * upper], [15.8 * lower, 15.8 * upper],
        [25.37 * lower, 25.37 * upper], [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
        [0.0056 * lower, 0.0056 * upper], [0.34 * lower, 0.34 * upper], [0.55 * lower, 0.55 * upper], [0.34 * lower, 0.34 * upper],
        [0.55 * lower, 0.55 * upper], [0.05 * lower, 0.05 * upper], [0.07 * lower, 0.07 * upper], [1.5 * lower, 1.5 * upper],
        [1.5 * lower, 1.5 * upper], # [6.8 * lower, 6.8 * upper], [-2 * 1.5, -2 * 0.5], [-6 * 1.5, -6 * 0.5],
        [0.73 * lower, 0.73 * upper], [0.04 * lower, 0.04 * upper],
        # cardio control
        [25 * lower, 25 * upper], [16.11 * lower, 16.11 * upper], [2.1 * lower, 2.1 * upper], [80 * lower, 80 * upper],
        [3.2 * lower, 3.2 * upper], [6.3 * lower, 6.3 * upper], [0.0675 * lower, 0.0675 * upper], [7.06 * lower, 7.06 * upper],
        [0.658 * lower, 0.658 * upper], [0.65 * lower, 0.65 * upper], [0.45 * lower, 0.45 * upper], [0.22 * lower, 0.22 * upper],
        [0.114 * lower, 0.114 * upper], [0.13 * lower, 0.13 * upper], [0.09 * lower, 0.09 * upper], [0.0162 * lower, 0.0162 * upper],
        [20 * lower, 20 * upper], [-0.0283 * upper, -0.0283 * lower], [5.5 * lower, 5.5 * upper], [-0.037 * upper, -0.037 * lower],
        [64.9 * lower, 64.9 * upper], [-0.437 * upper, -0.437 * lower], [1.9 * lower, 1.9 * upper], [-0.0008 * upper, -0.0008 * lower],
        [-0.68 * upper, -0.68 * lower], [-1.75 * upper, -1.75 * lower], [-1.1375 * upper, -1.1375 * lower], [-1.1375 * upper, -1.1375 * lower],
        [1 * lower, 1 * upper], [1.716 * lower, 1.716 * upper], [1.716 * lower, 1.716 * upper], [0.2 * lower, 0.2 * upper],
        [-0.2 * upper, -0.2 * lower], [-0.3997 * upper, -0.3997 * lower], [-0.3997 * upper, -0.3997 * lower], [-0.103 * upper, -0.103 * lower],
        [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper],
        [2.392 * lower, 2.392 * upper], [1.412 * lower, 1.412 * upper], [2.66 * lower, 2.66 * upper], [0.475 * lower, 0.475 * upper],
        [0.282 * lower, 0.282 * upper], [4.47 * lower, 4.47 * upper], [1.94 * lower, 1.94 * upper], [2.47 * lower, 2.47 * upper],
        [0.695 * lower, 0.695 * upper], [-28.29 * upper, -28.29 * lower], [-74.21 * upper, -74.21 * lower], [-28.29 * upper, -28.29 * lower],
        [-265.4 * upper, -265.4 * lower], [3.51 * lower, 3.51 * upper], [1.655 * lower, 1.655 * upper], [5.27 * lower, 5.27 * upper],
        #
        [2.49 * lower, 2.49 * upper], [(1 / 60) * lower, (1 / 60) * upper], [1 * lower, 1 * upper], [1.5 * lower, 1.5 * upper],
        [0.2 * lower, 0.2 * upper], [6 * lower, 6 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
        [45 * lower, 45 * upper], [30 * lower, 30 * upper], [30 * lower, 30 * upper], [3.6 * lower, 3.6 * upper],
        [13.32 * lower, 13.32 * upper], [13.32 * lower, 13.32 * upper], [53 * lower, 53 * upper], [6 * lower, 6 * upper],
        [6 * lower, 6 * upper], [40 * 0.9, 40 * 1.1], [47.78 * lower, 47.78 * upper], [2.52 * lower, 2.52 * upper],
        [11.76 * lower, 11.76 * upper], [92 * lower, 92 * 1.05], [112 * 0.9, 112 * upper], [1.4 * lower, 1.4 * upper],
        [12.3 * lower, 12.3 * upper], [0.835 * lower, 0.835 * upper], [29.27 * lower, 29.27 * upper], [3 * lower, 3 * upper],
        [45 * lower, 45 * upper], [11.76 * lower, 11.76 * upper], [-0.13 * upper, -0.13 * lower], [0.09 * lower, 0.09 * upper],
        [0.58 * lower, 0.58 * upper],  [20.9 * lower, 20.9 * upper], [92.8 * lower, 92.8 * upper], [10570 * lower, 10570 * upper],
        [-5.251 * upper, -5.251 * lower], [0.14 * lower, 0.14 * upper], [10 * lower, 10 * upper], [0.925 * lower, 0.925 * upper],
        [6.57 * lower, 6.57 * upper], [0.11 * lower, 0.11 * upper], [0.155 * lower, 0.155 * upper], [35 * lower, 35 * upper],
        [30 * lower, 30 * upper], [11.11 * lower, 11.11 * upper], [142.8 * lower, 142.8 * upper], [0.4 * lower, 0.4 * upper],
        [0.86 * lower, 0.86 * upper], [19.71 * lower, 19.71 * upper], [12660 * lower, 12660 * upper], [0.1555 * lower, 0.1555 * upper],
        [30 * lower, 30 * upper], [40 * lower, 40 * upper], [0.4266 * lower, 0.4266 * upper], [0.18 * lower, 0.18 * upper],
        [0.516 * lower, 0.516 * upper], [20 * lower, 20 * upper], [-1.87 * upper, -1.87 * lower],
        # added params
        [1000 * lower, 1000 * upper], [5000 * lower, 5000 * upper], [2 * lower, 2 * upper], [5 * lower, 5 * upper], [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper], [200 * lower, 200 * upper], [2 * lower, 2 * upper], [10 * lower, 10 * upper], [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper], [2000 * lower, 2000 * upper], [5 * lower, 5 * upper], [10 * lower, 10 * upper], [1.309 * lower, 1.309 * upper],
        [3000 * lower, 3000 * upper], [500 * lower, 500 * upper], [2 * lower, 2 * upper], [7 * lower, 7 * upper], [1.309 * lower, 1.309 * upper],
        [0.0000317 * lower, 0.0000317 * upper], [350 * lower, 350 * upper], [350 * lower, 350 * upper], [350 * lower, 350 * upper],
        [350 * lower, 350 * upper], [0.00134 * 0.9, 0.00134 * 1.1], [2.6 * 0.9, 2.6 * 1.1], [3.03e-5 * 0.9, 3.03e-5 * 1.1],
        [104 * lower, 104 * upper], [1 * lower, 1 * upper], [279.49 * lower, 279.49 * upper], [93.16 * lower, 93.16 * upper],
        [579.76 * lower, 579.76 * upper], [123 * lower, 123 * upper], [350 * lower, 350 * upper], [1.0 * lower, 1.0 * upper],
        [116.6775 * lower, 116.6775 * upper], [214 * lower, 214 * upper], [10 * lower, 10 * upper], [10 * lower, 10 * upper],
        [10 * lower, 10 * upper], [10 * lower, 10 * upper],

        [5027.6 * 0.8, 5027.6 * 1.2], [8 * lower, 8 * upper], [8 * lower, 8 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [20 * lower, 20 * upper],
        [20 * lower, 20 * upper], [20 * lower, 20 * upper], [20 * lower, 20 * upper], [286.4 * lower, 286.4 * upper],
        [607.8 * lower, 607.8 * upper], [190.95 * lower, 190.95 * upper], [1361.6 * lower, 1361.6 * upper], [20 * lower, 20 * upper],
        [30 * lower, 30 * upper], [2.076 * lower, 2.076 * upper], [0.8 * lower, 0.8 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [1.5 * lower, 1.5 * upper], [20 * lower, 20 * upper],
        [10 * lower, 10 * upper], [5 * lower, 5 * upper], [40 * lower, 40 * upper], [10 * lower, 10 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [5 * lower, 5 * upper], [5 * lower, 5 * upper],
        [5 * lower, 5 * upper], [5 * lower, 5 * upper], [2 * lower, 2 * upper], [0.2 * lower, 0.2 * upper],
        [4 * lower, 4 * upper], [3 * lower, 3 * upper], [0.014 * lower, 0.014 * upper], [0.011 * lower, 0.011 * upper],
        [1 * lower, 1 * upper], [2 * lower, 2 * upper], [3 * lower, 3 * upper], [2.5 * lower, 2.5 * upper],
        [20 * lower, 20 * upper], [0.09 * lower, 0.09 * upper], [50 * lower, 50 * upper], [0.25 * lower, 0.25 * upper],
        [0.25 * lower, 0.25 * upper], [50 * lower, 50 * upper],

        # further added params
        [4.9 * lower, 4.9 * upper], [1.5 * lower, 1.5 * upper], [0.3 * lower, 0.3 * upper], [26.6 * lower, 26.6 * upper],
        [0.5 * lower, 0.5 * upper], [1.2 * lower, 1.2 * upper], [30 * lower, 30 * upper], [80 * lower, 80 * upper],
        [0.05 * lower, 0.05 * upper], [0.15 * lower, 0.15 * upper], [0.3 * 0.8, 0.3 * 1.2], [0.9 * 0.95, 0.9 * 1.05],
        [0.0872665 * lower, 0.0872665 * upper], [1.3 * 0.8, 1.3 * 1.2], [2.0 * 0.8, 2.0 * 1.2], [280 * lower, 280 * upper], [40 * lower, 40 * upper]]
})


mask = Result_all[:,0] != 0
X = X_all[mask, :]
Result = Result_all[mask]
Result_states = Result_states[mask]

# try and see if emulator is better with or without filtering
# get the mean of the column
col_mean = Result.mean(axis=0)
col_std = Result.std(axis=0)
# 3 std to remove outliers
mask = (Result >= (col_mean - 3*col_std)) & (Result <= (col_mean + 3*col_std))
row_mask = mask.all(axis=1)
X = X[row_mask, :]
Result = Result[row_mask]
Result_states = Result_states[row_mask]


mask = np.ptp(X, axis=0) != 0  # ptp = max - min, 0 means all values identical
X = X[:, mask]

# idx = np.random.choice(len(Result), size=10000, replace=False)
X = X[:size,:]
Result = Result[:size]
Result_states = Result_states[:size]

# filter result_states so that theta_change_O2_sp, theta_change_O2_sv, P_n_current, MRTO2, MRTCO2, MRV, VE_integral are not needed for prediction
Initial_Condition_states = ['VT_pa', 'VT_pp', 'VT_pv', 'Q_pa', 'VT_la', 'VT_lv', 'VT_ra', 'VT_rv', 'VT_sv', 'VT_bv',
'VT_hv', 'VT_rmv', 'VT_amv', 'P_sp', 'P_sa', 'Q_sa', 'VT_vc', 'theta_ao', 'dtheta_ao_dt', 'theta_po', 'dtheta_po_dt',
'theta_mi', 'dtheta_mi_dt', 'theta_tr', 'dtheta_tr_dt', 'theta_change_O2_sp', 'theta_change_CO2_sp',
'theta_change_O2_sv', 'theta_change_CO2_sv', 'theta_change_O2_sh', 'theta_change_CO2_sh', 'P_tilda', 'f_ac', 'f_ap',
'R_ep_change', 'R_sp_change', 'R_rmp_n_change', 'R_amp_n_change', 'Vu_ev_change', 'Vu_sv_change', 'Vu_rmv_change',
'Vu_amv_change', 'Emax_lv_change', 'Emax_rv_change', 'Ts_change', 'Tv_change', 'xb_O2', 'xb_CO2', 'xh_O2', 'xh_CO2',
'Wh', 'xrm_O2', 'xrm_CO2', 'xam_O2', 'xM', 'x_met', 'P_n_current',

'Pd_1_O2', 'Pd_1_CO2', 'Pd_2_O2', 'Pd_2_CO2', 'Pd_3_O2', 'Pd_3_CO2', 'Pd_4_O2', 'Pd_4_CO2', 'Pd_5_O2', 'Pd_5_CO2',
'Pa_O2', 'Pa_CO2', 'dPa_O2_dt', 'dPa_CO2_dt', 'PA_O2', 'PA_CO2', 'PCSFCO2', 'MRTO2', 'MRTCO2', 'CTO2', 'CvtCO2', 'CBO2',
'CvbCO2', 'MRV', 'VE_integral']

# Unwanted states constant values: {"theta_change_O2_sp": 0.0, "theta_change_O2_sv": 0.0, "P_n_current": 92.0, "MRTO2": 0.00324166667, "MRTCO2": 0.00243333333, "MRV": 0.0, "VE_integral": 0.0}
Unwanted_states = {"theta_change_O2_sp", "theta_change_O2_sv", "P_n_current", "MRTO2", "MRTCO2", "MRV", "VE_integral"}
constant_values = {"theta_change_O2_sp": 0.0, "theta_change_O2_sv": 0.0, "P_n_current": 92.0, "MRTO2": 0.00324166667, "MRTCO2": 0.00243333333, "MRV": 0.0, "VE_integral": 0.0}
keep_mask = np.array([state not in Unwanted_states for state in Initial_Condition_states])
Result_states = Result_states[:, keep_mask]
kept_states = [s for s in Initial_Condition_states if s not in Unwanted_states]

###################################################################################
# Look at how good the state variable emulator is by giving it a test parameter set
state_gaussian_process = joblib.load("GaussianProcessMatern32_5000_rest_no_p_thor_state.joblib")
Parameters = {'beta2': 0.029304536132812503, 'C2': 84.85048828125, 'K2': 192.36867187500002, 'a2': 1.81420380859375, 'alpha2': 0.0565488154296875, 'dc': 0.017065429687499998, 'KCCO2': 310251.171875, 'GV_dead': 0.17573636718750002, 'KcCO2': 0.251555390625, 'KcMRV': 1.0685546875, 'KpCO2': 0.22856396484375, 'KpO2': 5.515578125e-09, 'V0_dead': 0.17181134765625, 'VA_rest': 0.06294916015625, 'Pmax': 85.29296875, 'Pmax_dot': 1142.7734375, 'E_rs': 25.009628906249997, 'R_rs': 2.6548867187500003, 'C_jp': 3.7773984375000005, 'C_sa': 0.26277343750000004, 'L_sa': 0.00022219140625000003, 'R_sa': 0.16980468750000002, 'C_amv': 4.973203125, 'C_bv': 5.717806640625, 'C_ev': 11.224609375, 'C_hv': 1.3777363281250001, 'C_rmv': 3.6214531249999995, 'C_sv': 35.332939453125, 'kr_am': 22.380853515625002, 'P_0': 3.2123144531250003, 'R_amv_n': 0.09978103515625, 'R_bv_n': 0.0889892578125, 'R_ev_n': 0.0380546875, 'R_hv_n': 0.25029375, 'R_rmv_n': 0.10983886718750001, 'R_sv_n': 0.04059023437499999, 'D1': 0.43918388671874997, 'K1_vc': 0.146923828125, 'Kr_vc': 8.927734375e-05, 'Rvc_n': 0.00212451171875, 'C_pa': 0.7355078125000001, 'C_pp': 13.161523437500001, 'C_pv': 28.407462890625, 'L_pa': 0.00020823046875000003, 'R_pa': 0.0221509765625, 'R_pp': 0.0752391796875, 'R_pv': 0.00483328125, 'Emax_la': 0.27458984375, 'P0_la': 0.657744140625, 'Emax_ra': 0.28242578125, 'P0_ra': 0.5511816406250001, 'KE_la': 0.055205078125, 'KE_ra': 0.060908203125000004, 'P0_lv': 1.75341796875, 'P0_rv': 1.39716796875, 'VT_n': 0.6220683593749999, 's': 0.0380859375, 'fab_o': 20.5126953125, 'fes_o': 18.410080078125, 'fes_inf': 2.26201171875, 'fes_max': 94.921875, 'fev_o': 3.019375, 'fev_inf': 5.61955078125, 'kes': 0.07982666015625, 'kev': 6.98140234375, 'Io_sh': 0.775848828125, 'Io_sp': 0.713857421875, 'Io_sv': 0.533408203125, 'Io_v': 0.17905078125, 'kcc_sh': 0.130899609375, 'kcc_sp': 0.12266210937500001, 'kcc_sv': 0.074935546875, 'kcc_v': 0.013792148437499999, 'Ysh_max': 18.38671875, 'Ysh_min': -0.02563029296875, 'Ysp_max': 5.46884765625, 'Ysp_min': -0.0359521484375, 'Ysv_max': 70.79423828125002, 'Ysv_min': -0.4109677734375, 'Yv_max': 1.74970703125, 'Yv_min': -0.00071078125, 'theta_v': -0.5470546875, 'Wb_sh': -1.9930175781250001, 'Wb_sp': -1.15327392578125, 'Wb_sv': -1.27724365234375, 'Wc_sh': 0.8052734375, 'Wc_sp': 2.00993203125, 'Wc_sv': 1.7853773437499998, 'Wc_v': 0.1905078125, 'Wp_sh': -0.16847656250000004, 'Wp_sp': -0.39977806640625, 'Wp_sv': -0.35496794921875, 'Wp_v': -0.1158951171875, 'Wt_sh': 0.432265625, 'Wt_sp': 0.356171875, 'Wt_sv': 0.393046875, 'Wt_v': 0.37726562500000005, 'Emax_lv0': 2.5597203124999997, 'Emax_rv0': 1.13594296875, 'fes_min': 2.93379296875, 'GEmax_lv': 0.5535791015625, 'GEmax_rv': 0.303976171875, 'GR_amp': 5.352650390625, 'GR_ep': 2.3139804687499996, 'GR_rmp': 2.401978515625, 'GR_sp': 0.6487119140625, 'GV_amv': -27.433564453125, 'GV_ev': -80.83382226562499, 'GV_rmv': -27.278853515625002, 'GV_sv': -243.88808593749997, 'R_amp0': 2.853931640625, 'R_ep0': 1.4762470703125001, 'R_rmp0': 6.182986328124999, 'R_sp0': 2.0985058593750003, 'AT': 0.015764973958333334, 'g_ccsh': 0.9685546875, 'g_ccsp': 1.7809570312499998, 'g_ccsv': 0.2024609375, 'kisc_sh': 7.119140625, 'kisc_sp': 1.6746093750000002, 'kisc_sv': 2.360546875, 'PO2_sh': 38.5927734375, 'PO2_sp': 33.533203125, 'PO2_sv': 26.431640625, 'theta_shn': 3.9086718750000005, 'theta_spn': 12.6930234375, 'theta_svn': 12.0062109375, 'x_sh': 45.722851562500004, 'x_sp': 6.376171875, 'x_sv': 5.3636718750000005, 'PaCO2_n': 38.62890625, 'f_ab_max': 48.53589453125, 'f_ab_min': 2.3236171875, 'k_ab': 13.269046875, 'P_n': 82.05654296875001, 'P_n_max': 112.10390625, 'f_acCO2_n': 1.1443359375, 'f_ac_max': 13.51318359375, 'f_ac_min': 0.7294833984375, 'k_ac': 29.229982421874997, 'K_H': 2.6583984375000003, 'PaO2_ac_n': 38.6103515625, 'G_ap': 10.374984375, 'GT_s': -0.12865429687500002, 'GT_v': 0.075041015625, 'T0': 0.6086601562499999, 'A': 21.01837890625, 'B': 77.23062499999999, 'C': 12578.712890625, 'D': -4.450017382812501, 'Cvb_O2_n': 0.13423046875000003, 'gb_O2': 10.087890625, 'MO2_bp': 0.8261767578125001, 'R_bpn': 6.3608378906250005, 'Cvh_O2_n': 0.10821679687500001, 'Cvrm_O2_n': 0.1595107421875, 'gh_O2': 30.1123046875, 'grm_O2': 29.513671875, 'Kh_CO2': 9.623603515625, 'Krm_CO2': 135.68789062500002, 'MO2_hpn': 0.38617187500000005, 'MO2_rmp': 0.7637539062500001, 'R_hpn': 19.436677734375, 'W_hn': 14709.83203125, 'Cvam_O2_n': 0.14660126953125, 'gam_O2': 31.482421875, 'gM': 32.3203125, 'Io_met': 0.5071707421874999, 'kmet': 0.20429296875, 'MO2_ampn': 0.44111953125000003, 'phi_max': 16.68359375, 'phi_min': -1.6899394531250003, 'Kp_ao': 1190.0390625, 'Kf_ao': 4260.7421875, 'Kb_ao': 1.689453125, 'Kv_ao': 5.8876953125, 'theta_ao_max': 1.4038513671875, 'Kp_mi': 1646.484375, 'Kf_mi': 489.55078125, 'Kb_mi': 1.876953125, 'Kv_mi': 6.156445312500001, 'theta_mi_max': 1.3015857421875, 'Kp_po': 1677.734375, 'Kf_po': 2145.703125, 'Kb_po': 4.1435546875, 'Kv_po': 9.857421875, 'theta_po_max': 1.4892431640625, 'Kp_tr': 2663.0859375, 'Kf_tr': 417.67578125, 'Kb_tr': 1.919921875, 'Kv_tr': 7.7341796875, 'theta_tr_max': 1.3956701171875, 'alpha_O2': 3.170619140625e-05, 'R_po': 366.748046875, 'R_mi': 398.740234375, 'R_tr': 413.779296875, 'R_ao': 310.283203125, 'C_O2_param1': 0.0013338496093750001, 'C_O2_param2': 2.82013671875, 'C_O2_param3': 2.8196162109375002e-05, 'PAMO2_nominal': 102.3546875, 'Vu_sa': 0.8447265625, 'Vu_bv': 283.802443359375, 'Vu_hv': 100.4927109375, 'Vu_jp': 535.032421875, 'Vu_vc': 115.0962890625, 'Vvc_max': 373.583984375, 'Vu_pa': 1.1556640625, 'Vu_pp': 124.12936376953124, 'Vu_pv': 248.064453125, 'Vu_la': 10.486328125, 'Vu_lv': 8.095703125, 'Vu_ra': 11.146484375, 'Vu_rv': 11.818359375, 'V_tot': 4592.594765625, 'tau_Emax_lv': 7.6609375, 'tau_Emax_rv': 8.2953125, 'tau_Ramp': 1.692578125, 'tau_Rep': 2.071484375, 'tau_Rrmp': 1.794140625, 'tau_Rsp': 1.852734375, 'tau_Vamv': 21.39453125, 'tau_Vev': 21.04296875, 'tau_Vrmv': 17.30078125, 'tau_Vsv': 20.23828125, 'Vu_amv0': 249.3134375, 'Vu_ev0': 515.0867578125, 'Vu_rmv0': 172.71278320312499, 'Vu_sv0': 1110.2890625, 'tau_cc': 21.03515625, 'tau_isc': 26.291015625, 'tau_p': 2.00747578125, 'tau_z': 0.86359375, 'tau_ac': 1.864453125, 'tau_ap': 1.695703125, 'tau_Ts': 1.6277343750000002, 'tau_Tv': 1.55654296875, 'tau_CO2': 16.51953125, 'tau_O2': 9.353515625, 'tau_w': 4.5732421875, 'tau_M': 41.2890625, 'tau_met': 9.255859375, 'DEmax_lv': 1.769921875, 'DEmax_rv': 2.256640625, 'DR_amp': 2.162890625, 'DR_ep': 2.055859375, 'DR_rmp': 1.7339843750000001, 'DR_sp': 1.808984375, 'DV_amv': 4.5615234375, 'DV_ev': 4.1923828125, 'DV_rmv': 5.6337890625, 'DV_sv': 5.2705078125, 'DT_s': 2.141015625, 'DT_v': 0.18699218750000002, 'Dmet': 4.38203125, 'Ta': 3.4048828124999995, 'KE_lv': 0.012799609375000002, 'KE_rv': 0.012540429687499998, 'T1': 0.9962890625, 'T2': 1.938671875, 'VL_CO2': 3.5080078125, 'VL_O2': 2.49169921875, 'KCSFCO2': 21.08203125, 'VB': 0.103025390625, 'tauMR': 46.337890625, 'VTCO2': 0.260693359375, 'VTO2': 0.214013671875, 'tau_MRV': 54.951171875, 'scale_param1': 4.16212890625, 'scale_param2': 1.76806640625, 'scale_param3': 0.25611328125, 'scale_param4': 23.9659765625, 'scale_param5': 0.49443359375, 'scale_param6': 1.088203125, 'scale_param7': 32.865234375, 'Pa_O2_lower': 90.859375, 'rise_time_atr': 0.050908203125, 'rise_time_ven': 0.141767578125, 'fall_time_ven': 0.27908203125, 'ahead1': 0.9325634765625, 'theta_min': 0.08557912041015625, 'r': 1.0272070312500001, 'l': 2.4240234375000003, 'V_nominal': 308.6015625, 'V_scale': 44.4453125}
Parameters = np.array(list(Parameters.values()))[mask]
param_values = torch.tensor(Parameters, dtype=torch.float32)

n_states = len(Initial_Condition_states)
Y_full = np.empty(n_states, dtype=float)

model_name = "GaussianProcessMatern32"
base_path = f"best_{model_name}"

for j, state_name in enumerate(kept_states):

    model_path = (
        f"{base_path}/"
        f"{model_name}_{state_name}_5000_rest_no_p_thor_state.joblib"
    )

    emulator = joblib.load(model_path)

    Result_tensor, _ = emulator.model.predict_mean_and_variance(param_values)

    # scalar prediction
    Y_full[Initial_Condition_states.index(state_name)] = (
        Result_tensor.detach().cpu().numpy().item()
    )

for i, state in enumerate(Initial_Condition_states):
    if state in constant_values:
        Y_full[i] = constant_values[state]

Initial_Conditions = {
    state: float(Y_full[i])
    for i, state in enumerate(Initial_Condition_states)
}

for k, v in Initial_Conditions.items():
    print(f"'{k}': {v},")
###################################################################################