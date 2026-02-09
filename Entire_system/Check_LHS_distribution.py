import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from SALib import ProblemSpec
import warnings
# Ignore only this specific FutureWarning from pandas
warnings.filterwarnings(
    "ignore",
    message=".*use_inf_as_na option is deprecated.*",
    category=FutureWarning
)

# X_all = np.load(f'LHCS_just_check.npy')[:182500, :]
# Result_all = np.load(f'Result_chunked.npy')
#
# mask = Result_all[:,0] != 0
# X = X_all[mask, :]
# Result = Result_all[mask]
#
# # get the mean of the column
# col_mean = Result.mean(axis=0)
# col_std = Result.std(axis=0)
# # 3 std to remove outliers
# mask = (Result >= (col_mean - 3*col_std)) & (Result <= (col_mean + 3*col_std))
# row_mask = mask.all(axis=1)
# X = X[row_mask]
# Result = Result[row_mask]
#
# lower = 0.5
# upper = 1.5
#
# sp = ProblemSpec({
#     'names': [
#         # gas
#         "beta2", "C2", "K2", "a2",
#         "alpha2", "dc", "KCCO2", "GV_dead",
#         # resp control
#         "KcCO2", "KcMRV", "KpCO2", "KpO2",
#         "V0_dead", "VA_rest", "Pmax", "Pmax_dot",
#         "E_rs", "R_rs",
#         # cardio
#         "C_jp", "C_sa", "L_sa", "R_sa",
#         "C_amv", "C_bv", "C_ev", "C_hv",
#         "C_rmv", "C_sv", "kr_am", "P_0",
#         "R_amv_n", "R_bv_n", "R_ev_n", "R_hv_n",
#         "R_rmv_n", "R_sv_n", "D1", "K1_vc",
#         "Kr_vc", "Rvc_n", "C_pa", "C_pp",
#         "C_pv", "L_pa", "R_pa", "R_pp",
#         "R_pv", "Emax_la", "P0_la", "Emax_ra",
#         "P0_ra", "KE_la", "KE_ra", "P0_lv",
#         "P0_rv", #"g_thor", "P_thormax_n", "P_thormin_n",
#         "VT_n", "s",
#         # cardio control
#         "fab_o", "fes_o", "fes_inf", "fes_max",
#         "fev_o", "fev_inf", "kes", "kev",
#         "Io_sh", "Io_sp", "Io_sv", "Io_v",
#         "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v",
#         "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min",
#         "Ysv_max", "Ysv_min", "Yv_max", "Yv_min",
#         "theta_v", "Wb_sh", "Wb_sp", "Wb_sv",
#         "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v",
#         "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v",
#         "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
#         "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv",
#         "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp",
#         "GR_sp", "GV_amv", "GV_ev", "GV_rmv",
#         "GV_sv", "R_amp0", "R_ep0", "R_rmp0",
#         #
#         "R_sp0", "AT", "g_ccsh", "g_ccsp",
#         "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv",
#         "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn",
#         "theta_spn", "theta_svn", "x_sh", "x_sp",
#         "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min",
#         "k_ab", "P_n", "P_n_max","f_acCO2_n",
#         "f_ac_max", "f_ac_min", "k_ac", "K_H",
#         "PaO2_ac_n", "G_ap", "GT_s", "GT_v",
#         "T0", "A", "B", "C",
#         "D", "Cvb_O2_n", "gb_O2", "MO2_bp",
#         "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2",
#         "grm_O2", "Kh_CO2", "Krm_CO2", "MO2_hpn",
#         "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n",
#         "gam_O2", "gM", "Io_met", "kmet",
#         "MO2_ampn", "phi_max", "phi_min",
#         # added params
#         "Kp_ao", "Kf_ao", "Kb_ao", "Kv_ao", "theta_ao_max",
#         "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max",
#         "Kp_po", "Kf_po", "Kb_po", "Kv_po", "theta_po_max",
#         "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max",
#         "alpha_O2", "R_po", "R_mi", "R_tr",
#         "R_ao", "C_O2_param1", "C_O2_param2", "C_O2_param3",
#         "PAMO2_nominal", "Vu_sa", "Vu_bv", "Vu_hv",
#         "Vu_jp", "Vu_vc", "Vvc_max", "Vu_pa",
#         "Vu_pp", "Vu_pv", "Vu_la", "Vu_lv",
#         "Vu_ra", "Vu_rv",
#
#         # "V_tot",
#         "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp",
#         "tau_Rep", "tau_Rrmp", "tau_Rsp", "tau_Vamv",
#         "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0",
#         "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc",
#         "tau_isc", "tau_p", "tau_z", "tau_ac",
#         "tau_ap", "tau_Ts", "tau_Tv", "tau_CO2",
#         "tau_O2", "tau_w", "tau_M", "tau_met",
#         "DEmax_lv", "DEmax_rv", "DR_amp", "DR_ep",
#         "DR_rmp", "DR_sp", "DV_amv", "DV_ev",
#         "DV_rmv", "DV_sv", "DT_s", "DT_v",
#         "Dmet", "Ta", "KE_lv", "KE_rv",
#         "T1", "T2", "VL_CO2", "VL_O2",
#         "KCSFCO2", "VB", "tauMR", "VTCO2",
#         "VTO2", "tau_MRV",
#
#         # further added
#         "scale_param1", "scale_param2", "scale_param3", "scale_param4",
#         "scale_param5", "scale_param6", "scale_param7", "Pa_O2_lower",
#         "rise_time_atr", "rise_time_ven", "fall_time_ven", "ahead1",
#         "theta_min", "r", "l", "V_nominal", "V_scale"
#     ],
#
#     'bounds': [
#         # gas
#         [0.03255 * 0.9, 0.03255 * 1.1], [87 * 0.9, 87 * 1.1], [194.4 * 0.9, 194.4 * 1.1], [1.819 * 0.9, 1.819 * 1.1],
#         [0.05591 * 0.9, 0.05591 * 1.1], [0.015 * lower, 0.015 * upper], [346000 * lower, 346000 * upper], [0.1698 * lower, 0.1698 * upper],
#         # resp control
#         [0.2332 * lower, 0.2332 * upper], [1 * lower, 1 * upper], [0.2025 * lower, 0.2025 * upper], [4.72e-09 * lower, 4.72e-09 * upper],
#         [0.1587 * lower, 0.1587 * upper], [0.0673 * lower, 0.0673 * upper], [100 * lower, 100 * upper], [1000 * lower, 1000 * upper],
#         [21.9 * lower, 21.9 * upper], [3.02 * lower, 3.02 * upper],
#         # cardio
#         [3.72 * lower, 3.72 * upper], [0.28 * lower, 0.28 * upper], [0.00022 * lower, 0.00022 * upper], [0.2 * lower, 0.2 * upper],
#         [4.4 * lower, 4.4 * upper], [5.71 * lower, 5.71 * upper], [10 * lower, 10 * upper], [1.57 * lower, 1.57 * upper],
#         [3.28 * lower, 3.28 * upper], [31.11 * lower, 31.11 * upper], [24.17 * lower, 24.17 * upper], [3.93 * lower, 3.93 * upper],
#         [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper], [0.224 * lower, 0.224 * upper],
#         [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper], [0.3855 * lower, 0.3855 * upper], [0.15 * lower, 0.15 * upper],
#         [0.0001 * lower, 0.0001 * upper], [0.0025 * lower, 0.0025 * upper], [0.76 * lower, 0.76 * upper], [15.8 * lower, 15.8 * upper],
#         [25.37 * lower, 25.37 * upper], [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
#         [0.0056 * lower, 0.0056 * upper], [0.30 * lower, 0.30 * upper], [0.55 * lower, 0.55 * upper], [0.34 * lower, 0.34 * upper],
#         [0.55 * lower, 0.55 * upper], [0.05 * lower, 0.05 * upper], [0.09 * lower, 0.09 * upper], [1.5 * lower, 1.5 * upper],
#         [1.5 * lower, 1.5 * upper], # [6.8 * lower, 6.8 * upper], [-2 * 1.5, -2 * 0.5], [-6 * 1.5, -6 * 0.5],
#         [0.73 * lower, 0.73 * upper], [0.04 * lower, 0.04 * upper],
#         # cardio control
#         [25 * lower, 25 * upper], [16.11 * lower, 16.11 * upper], [2.1 * lower, 2.1 * upper], [80 * lower, 80 * upper],
#         [3.2 * lower, 3.2 * upper], [6.3 * lower, 6.3 * upper], [0.0675 * lower, 0.0675 * upper], [7.06 * lower, 7.06 * upper],
#         [0.658 * lower, 0.658 * upper], [0.65 * lower, 0.65 * upper], [0.70 * lower, 0.70 * upper], [0.22 * lower, 0.22 * upper],
#         [0.114 * lower, 0.114 * upper], [0.13 * lower, 0.13 * upper], [0.09 * lower, 0.09 * upper], [0.0162 * lower, 0.0162 * upper],
#         [20 * lower, 20 * upper], [-0.0283 * upper, -0.0283 * lower], [5.5 * lower, 5.5 * upper], [-0.037 * upper, -0.037 * lower],
#         [64.9 * lower, 64.9 * upper], [-0.437 * upper, -0.437 * lower], [1.9 * lower, 1.9 * upper], [-0.0008 * upper, -0.0008 * lower],
#         [-0.68 * upper, -0.68 * lower], [-1.75 * upper, -1.75 * lower], [-1.1375 * upper, -1.1375 * lower], [-1.1375 * upper, -1.1375 * lower],
#         [1 * lower, 1 * upper], [1.716 * lower, 1.716 * upper], [1.716 * lower, 1.716 * upper], [0.2 * lower, 0.2 * upper],
#         [-0.2 * upper, -0.2 * lower], [-0.3997 * upper, -0.3997 * lower], [-0.3997 * upper, -0.3997 * lower], [-0.103 * upper, -0.103 * lower],
#         [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper],
#         [1.392 * lower, 1.392 * upper], [0.8 * lower, 0.8 * upper], [2.66 * lower, 2.66 * upper], [0.475 * lower, 0.475 * upper],
#         [0.282 * lower, 0.282 * upper], [4.47 * lower, 4.47 * upper], [1.94 * lower, 1.94 * upper], [2.47 * lower, 2.47 * upper],
#         [0.695 * lower, 0.695 * upper], [-28.29 * upper, -28.29 * lower], [-74.21 * upper, -74.21 * lower], [-28.29 * upper, -28.29 * lower],
#         [-265.4 * upper, -265.4 * lower], [3.51 * lower, 3.51 * upper], [1.655 * lower, 1.655 * upper], [5.27 * lower, 5.27 * upper],
#         #
#         [2.49 * lower, 2.49 * upper], [(1 / 60) * lower, (1 / 60) * upper], [1 * lower, 1 * upper], [1.5 * lower, 1.5 * upper],
#         [0.2 * lower, 0.2 * upper], [6 * lower, 6 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
#         [45 * lower, 45 * upper], [30 * lower, 30 * upper], [30 * lower, 30 * upper], [3.6 * lower, 3.6 * upper],
#         [13.32 * lower, 13.32 * upper], [13.32 * lower, 13.32 * upper], [53 * lower, 53 * upper], [6 * lower, 6 * upper],
#         [6 * lower, 6 * upper], [40 * 0.9, 40 * 1.1], [47.78 * lower, 47.78 * upper], [2.52 * lower, 2.52 * upper],
#         [11.76 * lower, 11.76 * upper], [80 * lower, 80 * upper], [112 * 0.9, 112 * upper], [1.4 * lower, 1.4 * upper],
#         [12.3 * lower, 12.3 * upper], [0.835 * lower, 0.835 * upper], [29.27 * lower, 29.27 * upper], [3 * lower, 3 * upper],
#         [45 * lower, 45 * upper], [11.76 * lower, 11.76 * upper], [-0.13 * upper, -0.13 * lower], [0.09 * lower, 0.09 * upper],
#         [0.8 * lower, 0.8 * upper],  [20.9 * lower, 20.9 * upper], [92.8 * lower, 92.8 * upper], [10570 * lower, 10570 * upper],
#         [-5.251 * upper, -5.251 * lower], [0.14 * lower, 0.14 * upper], [10 * lower, 10 * upper], [0.925 * lower, 0.925 * upper],
#         [6.57 * lower, 6.57 * upper], [0.11 * lower, 0.11 * upper], [0.155 * lower, 0.155 * upper], [35 * lower, 35 * upper],
#         [30 * lower, 30 * upper], [11.11 * lower, 11.11 * upper], [142.8 * lower, 142.8 * upper], [0.4 * lower, 0.4 * upper],
#         [0.86 * lower, 0.86 * upper], [19.71 * lower, 19.71 * upper], [12660 * lower, 12660 * upper], [0.1555 * lower, 0.1555 * upper],
#         [30 * lower, 30 * upper], [40 * lower, 40 * upper], [0.4266 * lower, 0.4266 * upper], [0.18 * lower, 0.18 * upper],
#         [0.516 * lower, 0.516 * upper], [20 * lower, 20 * upper], [-1.87 * upper, -1.87 * lower],
#         # added params
#         [1000 * lower, 1000 * upper], [5000 * lower, 5000 * upper], [2 * lower, 2 * upper], [5 * lower, 5 * upper], [1.309 * lower, 1.309 * upper],
#         [2000 * lower, 2000 * upper], [200 * lower, 200 * upper], [2 * lower, 2 * upper], [10 * lower, 10 * upper], [1.309 * lower, 1.309 * upper],
#         [2000 * lower, 2000 * upper], [2000 * lower, 2000 * upper], [5 * lower, 5 * upper], [10 * lower, 10 * upper], [1.309 * lower, 1.309 * upper],
#         [2000 * lower, 2000 * upper], [200 * lower, 200 * upper], [2 * lower, 2 * upper], [10 * lower, 10 * upper], [1.309 * lower, 1.309 * upper],
#         [0.0000317 * lower, 0.0000317 * upper], [350 * lower, 350 * upper], [350 * lower, 350 * upper], [350 * lower, 350 * upper],
#         [350 * lower, 350 * upper], [0.00134 * 0.9, 0.00134 * 1.1], [2.6 * 0.9, 2.6 * 1.1], [3.03e-5 * 0.9, 3.03e-5 * 1.1],
#         [104 * lower, 104 * upper], [1 * lower, 1 * upper], [279.49 * lower, 279.49 * upper], [93.16 * lower, 93.16 * upper],
#         [879.76 * lower, 879.76 * upper], [123 * lower, 123 * upper], [350 * lower, 350 * upper], [1.0 * lower, 1.0 * upper],
#         [116.6775 * lower, 116.6775 * upper], [214 * lower, 214 * upper], [20 * lower, 20 * upper], [60 * lower, 60 * upper],
#         [50 * lower, 50 * upper], [80 * lower, 80 * upper],
#
#         # [5027.6 * 0.8, 5027.6 * 1.2],
#         [8 * lower, 8 * upper], [8 * lower, 8 * upper], [2 * lower, 2 * upper],
#         [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [20 * lower, 20 * upper],
#         [20 * lower, 20 * upper], [20 * lower, 20 * upper], [20 * lower, 20 * upper], [286.4 * lower, 286.4 * upper],
#         [807.8 * lower, 807.8 * upper], [190.95 * lower, 190.95 * upper], [1661.6 * lower, 1661.6 * upper], [20 * lower, 20 * upper],
#         [30 * lower, 30 * upper], [2.076 * lower, 2.076 * upper], [0.8 * lower, 0.8 * upper], [2 * lower, 2 * upper],
#         [2 * lower, 2 * upper], [2 * lower, 2 * upper], [1.5 * lower, 1.5 * upper], [20 * lower, 20 * upper],
#         [10 * lower, 10 * upper], [5 * lower, 5 * upper], [40 * lower, 40 * upper], [10 * lower, 10 * upper],
#         [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
#         [2 * lower, 2 * upper], [2 * lower, 2 * upper], [5 * lower, 5 * upper], [5 * lower, 5 * upper],
#         [5 * lower, 5 * upper], [5 * lower, 5 * upper], [2 * lower, 2 * upper], [0.2 * lower, 0.2 * upper],
#         [4 * lower, 4 * upper], [0.3 * lower, 0.3 * upper], [0.014 * lower, 0.014 * upper], [0.015 * lower, 0.015 * upper],
#         [0.1 * lower, 0.1 * upper], [0.1 * lower, 0.1 * upper], [3 * lower, 3 * upper], [2.5 * lower, 2.5 * upper],
#         [20 * lower, 20 * upper], [0.01 * lower, 0.01 * upper], [50 * lower, 50 * upper], [0.25 * lower, 0.25 * upper],
#         [0.25 * lower, 0.25 * upper], [50 * lower, 50 * upper],
#
#         # further added params
#         [4.9 * lower, 4.9 * upper], [1.5 * lower, 1.5 * upper], [0.3 * lower, 0.3 * upper], [26.6 * lower, 26.6 * upper],
#         [0.5 * lower, 0.5 * upper], [1.2 * lower, 1.2 * upper], [30 * lower, 30 * upper], [80 * lower, 80 * upper],
#         [0.05 * lower, 0.05 * upper], [0.15 * lower, 0.15 * upper], [0.3 * 0.8, 0.3 * 1.2], [0.9 * 0.95, 0.9 * 1.05],
#         [0.0872665 * lower, 0.0872665 * upper], [1.5 * 0.67, 1.5 * upper], [2.0 * 0.5, 2.0 * upper], [380 * lower, 380 * upper], [40 * lower, 40 * upper]]
# })
#
#
# # No P_thor: 52 parameters contribute 90% sensitivity for 30 targets
# subset_vars = {'C_jp','C_pa','C_pp','C_pv','C_sa','Cvam_O2_n','Cvrm_O2_n',
# 'Emax_la','Emax_lv0','Emax_ra','Emax_rv0','GT_s','GT_v','G_ap','Io_met','Io_sv',
# 'K1_vc','K2','KE_la','KE_lv','KE_ra','KE_rv', 'Kp_po','Kp_tr','Kv_po','Kv_tr',
# 'P0_la','P0_lv','P0_ra','P0_rv','P_n','R_ep0','R_pa','R_po','R_pp','R_sa','R_sp0',
# 'T0','V_nominal','V_scale','Vu_amv0','Vu_bv','Vu_ev0','Vu_jp','Vu_pv','Vu_rmv0','Vu_sv0',
# 'Wb_sh','Wb_sp','Wb_sv','Wp_v','a2','f_ab_max','fab_o','fall_time_ven','fes_inf','fes_min','fes_o',
# 'fev_inf','fev_o','k_ab','kcc_sv','kes','kmet','l','r','rise_time_atr','rise_time_ven',
# 'theta_spn','theta_svn','theta_tr_max','theta_v'
# }
#
# output_names = [
#     "Heart Rate", "Systolic Pressure", "Diastolic Pressure", "EDV", "ESV",
#     "Max RV Volume", "Min RV Volume", "Max RV Pressure", "Min RV Pressure",
#     "Min RA Volume", "Max RA Volume", "Min RA Pressure A descent", "Max RA Pressure Atrial contraction",
#     "Max RA Pressure Tricuspid Opening", "Min RA Pressure V descent",
#     "Min LA Volume", "Max LA Volume", "Min LA Pressure A descent", "Max LA Pressure Atrial contraction",
#     "Max LA Pressure Tricuspid Opening", "Min LA Pressure V descent",
#     "LA EDV", "RA EDV", "LV Pressure Deriv", "RV Pressure Deriv", "Tidal Volume", "Minute Ventilation",
#     "Cardiac Output", "PaO2", "PaCO2", "Percentage Volume Change",
# #    "Stroke Volume", "Ejection Fraction"
# ]
#
# # rest, edited minute ventilation, diastolic pressure, Systolic Pressure, rv pressure deriv, cardiac output, PaO2, and min rv volume for now
# # observation = {"Heart Rate": (1.1, 0.1), "Systolic Pressure": (100, 20), "Diastolic Pressure": (40, 25), "EDV": (163, 23),
# # "ESV": (50, 30), "Max RV Volume": (186, 51), "Min RV Volume": (80, 29), "Max RV Pressure": (25, 20),
# # "Min RV Pressure": (2, 10), "Min RA Volume": (50, 5), "Max RA Volume": (93, 16), "Min RA Pressure A descent": (2, 1),
# # "Max RA Pressure Atrial contraction": (7, 2), "Max RA Pressure Tricuspid Opening": (7, 2),
# # "Min RA Pressure V descent": (2, 1), "Min LA Volume": (22, 5), "Max LA Volume": (72, 12),
# # "Min LA Pressure A descent": (2, 1), "Max LA Pressure Atrial contraction": (7, 2),
# # "Max LA Pressure Tricuspid Opening": (7, 2), "Min LA Pressure V descent": (2, 1), "LA EDV": (46, 5), "RA EDV": (47, 5),
# # "LV Pressure Deriv": (1600, 350), "RV Pressure Deriv": (500, 250), "Tidal Volume": (0.5, 0.1),
# # "Minute Ventilation": (10.0, 10.5), "Cardiac Output": (40, 14.2), "PaO2": (100, 14.5), "PaCO2": (40, 20), "Percentage Volume Change": (0.2, 0.1)}
#
#
# observation = {"Heart Rate": (1.1, 0.1), "Systolic Pressure": (105, 5), "Diastolic Pressure": (70, 3), "EDV": (163, 23),
# "ESV": (50, 20), "Max RV Volume": (186, 41), "Min RV Volume": (52, 29), "Max RV Pressure": (25, 10),
# "Min RV Pressure": (2, 1), "Min RA Volume": (50, 5), "Max RA Volume": (93, 16), "Min RA Pressure A descent": (2, 1),
# "Max RA Pressure Atrial contraction": (7, 2), "Max RA Pressure Tricuspid Opening": (7, 2),
# "Min RA Pressure V descent": (2, 1), "Min LA Volume": (25, 5), "Max LA Volume": (72, 12),
# "Min LA Pressure A descent": (2, 1), "Max LA Pressure Atrial contraction": (7, 2),
# "Max LA Pressure Tricuspid Opening": (7, 2), "Min LA Pressure V descent": (2, 1), "LA EDV": (45, 5), "RA EDV": (65, 5),
# "LV Pressure Deriv": (1600, 550), "RV Pressure Deriv": (500, 250), "Tidal Volume": (0.5, 0.1),
# "Minute Ventilation": (6.5, 2.5), "Cardiac Output": (85, 24.2), "PaO2": (90, 14.5), "PaCO2": (40, 12), "Percentage Volume Change": (0.2, 0.1)}
#
# obs_mean = np.array([observation[name][0] for name in output_names])
# obs_std  = np.array([observation[name][1] for name in output_names])
#
# k = 3
# lower = obs_mean - k * obs_std
# upper = obs_mean + k * obs_std
#
# within_bounds = (Result >= lower) & (Result <= upper)
# # rows that satisfy all outputs simultaneously
# valid_row_mask = within_bounds.all(axis=1)
#
# for i, name in enumerate(output_names):
#     frac_ok = np.mean(within_bounds[:, i])
#     print(f"{name:35s}: {frac_ok*100:6.2f}% within range")
#
# n_valid = np.sum(valid_row_mask)
#
# X_valid = X[valid_row_mask]
# Result_valid = Result[valid_row_mask]
#
# valid_parameter_set = X_valid[5]
#
# print(f"Number of valid rows: {n_valid}")
#
# if n_valid == 0:
#     print("❌ No samples satisfy all observation constraints.")
# else:
#     print("✅ At least one valid sample exists.")
#
# Parameters = {}
#
# subset_idx = 0
#
# for name, (low, high) in zip(sp['names'], sp['bounds']):
#     Parameters[name] = float(valid_parameter_set[subset_idx])
#     subset_idx += 1
#
# print(Parameters)
#
# cols = 5
# rows = int(np.ceil(len(output_names) / cols))
#
# fig, axes = plt.subplots(rows, cols, figsize=(18, 15))
# axes = axes.flatten()
# for i, ax in enumerate(axes):
#     if i < Result.shape[1]:
#         sns.kdeplot(Result[:, i], fill=True, ax=ax)
#         ax.set_title(output_names[i], fontsize=10, pad=1)
#     else:
#         ax.axis("off")
# plt.tight_layout()
# plt.show()


# change
size = 5000

# change
X_all = np.load(f'LHCS_20000_X_rest_no_Pthor.npy')
Result_all = np.load(f'LHCS_20000_Result_rest_no_Pthor.npy')

# mask = Result_all[:,0] != 0
# X = X_all[mask, :]
# Result = Result_all[mask]
#
# # get the mean of the column
# col_mean = Result.mean(axis=0)
# col_std = Result.std(axis=0)
# # 3 std to remove outliers
# mask = (Result >= (col_mean - 3*col_std)) & (Result <= (col_mean + 3*col_std))
# row_mask = mask.all(axis=1)
# X = X[row_mask, :]
# Result = Result[row_mask]

# # max LV systolic
# fig, ax1 = plt.subplots()
# # ax1.scatter(X_all[:,19], Result_all[:,1], marker='o', alpha=0.2, s=10, label="C_sa")
# # ax1.scatter(X_all[:,21], Result_all[:,1], marker='o', alpha=0.2, s=10, label="R_sa")
# # ax1.scatter(X_all[:,277], Result_all[:,1], marker='o', alpha=0.2, s=10, label="rise_time_ven")
# ax1.scatter(X_all[:,63], Result_all[:,1], marker='o', alpha=0.2, s=10, label="kes")
#
#
# ax1.legend(loc="upper left")
# plt.show()


# HR
fig, ax1 = plt.subplots()
# ax1.scatter(X_all[:,19], Result_all[:,1], marker='o', alpha=0.2, s=10, label="C_sa")
# ax1.scatter(X_all[:,44], Result_all[:,7], marker='o', alpha=0.2, s=10, label="R_pa")
# ax1.scatter(X_all[:,185], Result_all[:,7], marker='o', alpha=0.2, s=10, label="Kv_po")
ax1.scatter(X_all[:,44], Result_all[:,0], marker='o', alpha=0.2, s=10, label="GT_s")

ax1.legend(loc="upper left")
plt.show()

# max RV P
fig, ax1 = plt.subplots()
# ax1.scatter(X_all[:,19], Result_all[:,1], marker='o', alpha=0.2, s=10, label="C_sa")
# ax1.scatter(X_all[:,44], Result_all[:,7], marker='o', alpha=0.2, s=10, label="R_pa")
# ax1.scatter(X_all[:,185], Result_all[:,7], marker='o', alpha=0.2, s=10, label="Kv_po")
ax1.scatter(X_all[:,67], Result_all[:,5], marker='o', alpha=0.2, s=10, label="Io_sv")

ax1.legend(loc="upper left")
plt.show()


# # max RA V
# fig, ax1 = plt.subplots()
# # ax1.scatter(X_all[:,283], Result_all[:,11], marker='o', alpha=0.2, s=10, label="V_nominal")
# # ax1.scatter(X_all[:,281], Result_all[:,11], marker='o', alpha=0.2, s=10, label="r")
# # ax1.scatter(X_all[:,257], Result_all[:,11], marker='o', alpha=0.2, s=10, label="KE_rv")
# # ax1.scatter(X_all[:,54], Result_all[:,11], marker='o', alpha=0.2, s=10, label="P0_rv")
# # ax1.scatter(X_all[:,277], Result_all[:,11], marker='o', alpha=0.2, s=10, label="kcc_sv")
# # ax1.scatter(X_all[:,204], Result_all[:,11], marker='o', alpha=0.2, s=10, label="Vu_jp")
# ax1.scatter(X_all[:,41], Result_all[:,11], marker='o', alpha=0.2, s=10, label="C_pp")
#
#
# plt.show()

# max RA V
fig, ax1 = plt.subplots()
# ax1.scatter(X_all[:,281], Result_all[:,10], marker='o', alpha=0.2, s=10, label="r")
# ax1.scatter(X_all[:,283], Result_all[:,10], marker='o', alpha=0.2, s=10, label="V_nominal")
ax1.scatter(X_all[:,257], Result_all[:,10], marker='o', alpha=0.2, s=10, label="KE_rv")
# ax1.scatter(X_all[:,228], Result_all[:,10], marker='o', alpha=0.2, s=10, label="Vu_sv0")
# ax1.scatter(X_all[:,52], Result_all[:,10], marker='o', alpha=0.2, s=10, label="KE_ra")
# ax1.scatter(X_all[:,278], Result_all[:,10], marker='o', alpha=0.2, s=10, label="fall_time_ven")
# ax1.scatter(X_all[:,208], Result_all[:,10], marker='o', alpha=0.2, s=10, label="Vu_pp")
# ax1.scatter(X_all[:,54], Result_all[:,10], marker='o', alpha=0.2, s=10, label="P0_rv")
# ax1.scatter(X_all[:,50], Result_all[:,10], marker='o', alpha=0.2, s=10, label="P0_ra")

# max RV volume
# ax1.scatter(X_all[:,204], Result_all[:,5], marker='o', alpha=0.2, s=10, label="Vu_jp")
# ax1.scatter(X_all[:,277], Result_all[:,10], marker='o', alpha=0.2, s=10, label="rise_time_ven")
# ax1.scatter(X_all[:,145], Result_all[:,5], marker='o', alpha=0.2, s=10, label="T0")
# ax1.legend(loc="upper left")
plt.show()

# max RV P
fig, ax1 = plt.subplots()
# ax1.scatter(X_all[:,228], Result_all[:,7], marker='o', alpha=0.2, s=10, label="Vu_sv0")
# ax1.scatter(X_all[:,277], Result_all[:,7], marker='o', alpha=0.2, s=10, label="rise_time_ven")
# ax1.scatter(X_all[:,226], Result_all[:,7], marker='o', alpha=0.2, s=10, label="Vu_ev0")
ax1.scatter(X_all[:,204], Result_all[:,7], marker='o', alpha=0.2, s=10, label="Vu_jp")
# ax1.scatter(X_all[:,98], Result_all[:,7], marker='o', alpha=0.2, s=10, label="Emax_rv0")
# ax1.scatter(X_all[:,67], Result_all[:,7], marker='o', alpha=0.2, s=10, label="Io_sv")
# ax1.scatter(X_all[:,84], Result_all[:,7], marker='o', alpha=0.2, s=10, label="Wb_sv")

ax1.legend(loc="upper left")
plt.show()