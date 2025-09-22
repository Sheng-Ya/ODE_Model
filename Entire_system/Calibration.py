import os

import joblib
from SALib import ProblemSpec
from SALib.plotting.bar import plot as barplot
from SALib.analyze import sobol
from SALib.analyze.sobol import analyze
# from SALib.sample import saltelli
from SALib.sample.sobol import sample
from SALib.test_functions import Ishigami
import matplotlib.pyplot as plt
from autoemulate.logging_config import _configure_logging
import numpy as np
from autoemulate.compare import AutoEmulate
from sklearn.model_selection import KFold

X = np.load('All_params_LHCS_200000_X_sample_HR_Plv_Prv_Vlv_Vrv_rest.npy') # size [N, 299]
Result_all = np.load('All_params_LHCS_200000_Result_HR_Plv_Prv_Vlv_Vrv_rest.npy') # size [N, 9]

Stroke_Volume = Result_all[:, 3] - Result_all[:, 4]
Ejection_fraction = (Stroke_Volume / Result_all[:, 3]) * 100

Result_all = np.column_stack((Result_all, Stroke_Volume))
Result_all = np.column_stack((Result_all, Ejection_fraction))

mask = Result_all[:,0] != 0

X = X[mask, :]
Result = Result_all[mask, :]

# choose which results (column) to look at
Result_cols = ["Heart Rate", "Systolic Pressure", "Diastolic Pressure", "EDV", "ESV", "Max RV Volume", "Min RV Volume",
               "Max RV Pressure", "Min RV Pressure", "Stroke Volume", "Ejection Fraction"]
Result = Result[:, 0] # Heart rate here

## EMULATION
idx = np.random.choice(len(Result), size=30000, replace=False)
X = X[idx,:]
Result = Result[idx]

ae = AutoEmulate()
ae.logger = _configure_logging()
rbf_final_loaded = ae.load("rbf_final_hyper_30000_120s")


















# variables= [
#         "beta2", "C2", "K2", "a2", "alpha2", "dc", "KCCO2",
#         # "MRBCO2",
#         "GV_dead",
#         # "Kbg",
#         "KcCO2", "KcMRV", "KpCO2", "KpO2", "V0_dead", "VA_rest", "Pmax",
#         "Pmax_dot", "E_rs", "R_rs",
#         # cardio
#         "C_jp", "C_sa", "L_sa", "R_sa", "C_amv", "C_bv",
#         "C_ev", "C_hv", "C_rmv", "C_sv", "kr_am", "P_0", "R_amv_n", "R_bv_n",
#         "R_ev_n", "R_hv_n", "R_rmv_n", "R_sv_n", "D1", "K1_vc", "Kr_vc", "Rvc_n",
#         "C_pa", "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv", "Emax_la", "P0_la", "Emax_ra",
#         "P0_ra", "KE_la", "KE_ra", "P0_lv", "P0_rv", "g_abd", "g_thor", "P_abdmax_n", "P_abdmin_n",
#         "P_thormax_n", "P_thormin_n",
#         "VT_n", "A_im", "Tc", "T_im", "s",
#         # cardio control
#         "fab_o", "fes_o", "fes_inf", "fes_max", "fev_o", "fev_inf",
#         "kes", "kev", "Io_sh", "Io_sp", "Io_sv", "Io_v", "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v", "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min",
#         "Ysv_max", "Ysv_min", "Yv_max", "Yv_min", "theta_v", "Wb_sh", "Wb_sp", "Wb_sv", "Wc_sh", "Wc_sp",
#         "Wc_sv", "Wc_v", "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
#         "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
#         "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "AT", "g_ccsh", "g_ccsp",
#         "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv", "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn", "theta_spn",
#         "theta_svn", "x_sh", "x_sp", "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min", "k_ab", "P_n", "P_n_max",
#         "f_acCO2_n", "f_ac_max",
#         "f_ac_min", "k_ac", "K_H", "PaO2_ac_n", "G_ap", "GT_s", "GT_v", "T0", "A", "B",
#         "C", "D", "Cvb_O2_n", "gb_O2", "MO2_bp", "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2", "grm_O2",
#         "Kh_CO2", "Krm_CO2", "MO2_hpn", "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n", "gam_O2", "gM", "Io_met", "kmet",
#         "MO2_ampn", "phi_max", "phi_min",
#         # added params
#         "Kp_ao", "Kf_ao", "Kb_ao", "Kv_ao", "theta_ao_max", "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max", "Kp_po",
#         "Kf_po", "Kb_po", "Kv_po", "theta_po_max", "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max", "alpha_O2",
#         "R_po", "R_mi", "R_tr", "R_ao", "C_O2_param1", "C_O2_param2", "C_O2_param3", "PAMO2_nominal",
#         "Vu_sa", "V_tot", "Vu_bv", "Vu_hv", "Vu_jp", "Vu_vc",
#         "Vvc_max", "Vvc_min", "Vu_pa", "Vu_pp", "Vu_pv", "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "tau_Emax_lv",
#         "tau_Emax_rv", "tau_Ramp", "tau_Rep", "tau_Rrmp", "tau_Rsp", "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv",
#         "Vu_amv0", "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc", "tau_isc", "tau_p", "tau_z", "tau_ac", "tau_ap",
#         "tau_Ts", "tau_Tv", "tau_CO2", "tau_O2", "tau_w", "tau_M", "tau_met", "DEmax_lv", "DEmax_rv", "DR_amp",
#         "DR_ep", "DR_rmp", "DR_sp", "DV_amv", "DV_ev", "DV_rmv", "DV_sv", "DT_s", "DT_v", "Dmet", "Fi_CO2",
#         "Fi_O2", "Ta", "KE_lv", "KE_rv", "T1", "T2", "VL_CO2", "VL_O2", "KCSFCO2", "VB", "tauMR", "VTCO2", "VTO2", "tau_MRV",
#         "scale_param1", "scale_param2", "scale_param3", "scale_param4",
#         "scale_param5", "scale_param6", "scale_param7", "scale_param8",
#         "shift_param1", "shift_param2", "shift_param3", "shift_param4",
#         "Pa_O2_lower", "rise_time_atr", "fall_time_atr", "rise_time_ven",
#         "fall_time_ven", "ahead1", "ahead2"
#     ]
#
# plots_per_figure = 9   # match number of variables
# rows, cols = 3, 3      # grid for 9 plots
#
# for i in range(0, X.shape[1], plots_per_figure):   # loop over parameter chunks
#     fig, axes = plt.subplots(rows, cols, figsize=(24, 9))
#     axes = axes.flatten()
#
#     for k, j in enumerate(range(i, min(i + plots_per_figure, X.shape[1]))):
#         hb = axes[k].hexbin(X[:, j], Result, gridsize=50, cmap="viridis")
#         fig.colorbar(hb, ax=axes[k], label="Density")
#         axes[k].set_title(variables[j], fontsize=10)
#         axes[k].set_xlabel(variables[j], fontsize=8)
#         axes[k].set_ylabel("Heart Rate", fontsize=8)
#         axes[k].grid(True)
#
#     # hide any unused axes (in case <20 in last figure)
#     for ax in axes[k+1:]:
#         ax.set_visible(False)
#
#     plt.tight_layout()
#     plt.show()
