import os
import joblib
import pandas as pd
import seaborn as sns
import torch
from SALib import ProblemSpec
from autoemulate.calibration.history_matching import HistoryMatchingWorkflow
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from AutoEmulate_Simulator import Cardiopulmonary

from SALib.plotting.bar import plot as barplot
from SALib.analyze import sobol
from Entire_system.sobol_analyze_NIMP import analyze_NIMP
from scipy.special import binom
from scipy.stats import norm

from SALib.analyze.sobol import analyze
from SALib.util import scale_samples
# from SALib.sample import saltelli
from SALib.sample.sobol import sample
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')  # non-interactive backend
import numpy as np
from autoemulate import AutoEmulate
import warnings
# Ignore only this specific FutureWarning from pandas
warnings.filterwarnings(
    "ignore",
    message=".*use_inf_as_na option is deprecated.*",
    category=FutureWarning
)
from sklearn.model_selection import KFold

# A = AutoEmulate.list_emulators()
# print(A)
# X_all = np.load('LHCS/EDP_LHCS_200000_X_sample_rest.npy')[:150000]
# Result_all = np.load('LHCS/Results_LHCS_EDP_150000.npy')[:,0]

# change
Variable = "P_sys"
size = 5000
observation={"P_sys": (105, 5)}
calc_second_order = False

X_all = np.load(f'NROY_Points_rest.npy', allow_pickle=True)
Param_ranges = np.load(f'NROY_Params_rest.npy', allow_pickle=True).item()
GaussianProcess_final= joblib.load(f"GaussianProcessMatern32_10000_rest_no_p_thor.joblib")

# HR, EDP, ESP, Max RA pressure has
# [0.03255 * lower, 0.03255 * upper], [87 * 0.9, 87 * 1.1],
# [194.4 * 0.9, 194.4 * 1.1], [1.819 * 0.9, 1.819 * 1.1],
# [0.05591 * lower, 0.05591 * upper], [0.015 * lower, 0.015 * upper],

# Max RV pressure, EDV, PaO2, Minute Ventilation has
# [0.03255 * 0.9, 0.03255 * 1.1], [87 * 0.9, 87 * 1.1],
# [194.4 * 0.9, 194.4 * 1.1], [1.819 * 0.9, 1.819 * 1.1],
# [0.05591 * 0.9, 0.05591 * 1.1], [0.015 * lower, 0.015 * upper],

# change exercise/rest
lower = 0.5
upper = 1.5
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

# change
# HR
# subset_vars = ['T0', 'V_tot', 'P_n', 'fev_o', 'GT_v', 'GT_s', 'C2', 'C_O2_param1', 'Fi_O2',
#  'Vu_sv0', 'fes_o', 'fab_o', 'kes', 'Wb_sh', 'K2', 'k_ab', 'f_acCO2_n']
# exercise
# subset_vars = ['T0', 'GT_s', 'GT_v', 'fev_o', 'Fi_O2', 'AT', 'V_tot', 'Yv_max', 'Io_sh', 'R_rs',
#  'E_rs', 'Wp_v', 'G_ap', 'P_n_max', 'Ysh_max']

# SP
subset_vars = {'C2','C_O2_param1','C_sa','Cvam_O2_n','E_rs','Emax_la','Emax_lv0','Emax_rv0','GT_v','Io_sv','K2',
               'KE_la','KE_lv','KE_ra','KE_rv','KcCO2','MO2_bp','P0_la','P0_lv','P0_rv','P_n','PaCO2_n','R_pa',
               'R_rs','R_sa','T0','V0_dead','VA_rest','V_nominal','V_scale','V_tot','Vu_ev0','Vu_jp','Vu_pp',
               'Vu_rv','Vu_sv0','Wb_sh','Wb_sp','Wp_v','a2','ahead1','f_ab_max','fall_time_ven','fes_inf','fes_min',
               'fes_o','fev_inf','fev_o','kes','l','r','rise_time_ven'}
# exercise
# subset_vars = ['V_tot', 'Vu_sv0', 'GV_sv', 'R_rs', 'G_ap', 'R_sa', 'fes_o', 'P_n', 'Fi_O2',
#  'E_rs', 'fab_o', 'C_pv', 'rise_time_ven', 'GT_v', 'Vu_ev0', 'f_acCO2_n', 'C_sv',
#  'Vu_jp', 'fall_time_ven', 'T0', 'Wc_v', 'C_O2_param1', 'Kv_mi', 'k_ab',
#  'V0_dead', 'C_pp', 'Kp_mi', 'GV_dead', 'Wb_sh', 'fev_inf', 'Kv_tr', 'fev_o',
#  'Wp_v', 'Ysh_max', 'PaO2_ac_n', 'kev', 'theta_v', 'AT', 'tauMR', 'VA_rest',
#  'P_n_max', 'GT_s', 'R_pv', 'f_ab_max', 'k_ac', 'GR_amp', 'f_ac_max', 'Yv_max',
#  'Io_met', 'theta_mi_max', 'KE_lv', 'Kp_tr', 'Io_sh', 'MO2_bp', 'KcCO2', 'Tc',
#  'Vu_amv0', 'theta_tr_max', 'phi_max', 'Vu_bv', 'kes', 'PaCO2_n', 'f_ac_min']

# # DP
# subset_vars = ['Wp_v', 'fab_o', 'G_ap', 'theta_v', 'Fi_O2', 'a2', 'Vu_ev0', 'C2', 'C_O2_param1',
#                'C_pp', 'Vu_jp', 'R_bpn', 'T0', 'fes_o', 'C_pv', 'PaCO2_n', 'R_sp0', 'V_tot',
#                'GV_sv', 'kes', 'K2', 'P_n', 'k_ab', 'Wb_sh', 'Kp_tr', 'Cvrm_O2_n', 'fev_inf',
#                'theta_mi_max', 'Kv_mi', 'kev', 'fev_o', 'KE_lv', 'Emax_lv0', 'MO2_bp', 'R_pv',
#                'f_acCO2_n', 'GT_s', 'Kp_mi', 'theta_spn', 'theta_shn', 'Wc_v', 'kcc_sv',
#                'Vu_bv', 'Vu_sv0', 'E_rs', 'Kv_tr', 'Cvb_O2_n', 'C_sv', 'PaO2_ac_n', 'C_jp',
#                'Wb_sp', 'f_ac_max', 'Io_met', 'GT_v', 'f_ab_min', 'Io_sv', 'V0_dead', 'Vu_vc',
#                'GR_ep', 'fall_time_ven', 'f_ab_max', 'KcCO2', 'Cvam_O2_n', 'k_ac',
#                'theta_tr_max', 'Wb_sv', 'phi_min', 'kmet', 'Vu_rmv0', 'VA_rest', 'KE_rv',
#                'C_O2_param2', 'P0_lv', 'Vu_amv0', 'R_ep0', 'Rvc_n', 'fes_inf', 'g_ccsh',
#                'theta_svn', 'fes_min', 'GV_dead', 'R_mi', 'MO2_rmp']
# exercise
# subset_vars = ['k_ab', 'Wb_sh', 'theta_v', 'Wp_v', 'G_ap', 'kev', 'fev_inf', 'Io_sh', 'AT',
#  'fab_o', 'phi_max', 'C_O2_param1', 'P_n', 'MO2_bp', 'f_ac_max', 'tauMR',
#  'P_n_max', 'Kv_mi', 'f_ab_max', 'GT_s', 'GT_v', 'V_tot', 'VA_rest', 'Yv_max',
#  'Io_met', 'fev_o', 'PaO2_ac_n', 'fes_o', 'R_pv', 'GV_sv', 'Io_sv', 'Kp_mi',
#  'E_rs', 'Fi_O2', 'Vu_ev0', 'V0_dead', 'Vu_amv0', 'KcCO2', 'C_pp', 'Vu_sv0',
#  'theta_mi_max', 'GR_amp', 'theta_spn', 'Vu_jp', 'f_acCO2_n', 'fall_time_ven',
#  'k_ac', 'f_ab_min', 'Ysh_max', 'Kv_tr', 'f_ac_min', 'C_sv', 'kes', 'KE_lv',
#  'R_bpn', 'Io_v', 'Wc_v', 'T0', 'C_pv', 'GV_dead', 'Vu_bv', 'KE_rv', 'Cvam_O2_n',
#  'Cvrm_O2_n', 'Vu_vc', 'scale_param4', 'R_rs', 'theta_tr_max', 'Cvb_O2_n',
#  'theta_shn', 'fes_min', 'kcc_sh', 'Kp_tr', 'R_sp0', 'P0_lv', 'Wb_sp', 'MO2_ampn',
#  'Wc_sh']

# Max RV Pressure: 46 parameters contribute 90% sensitivity
# subset_vars = ['V_tot', 'PaCO2_n', 'C2', 'R_rs', 'a2', 'V0_dead', 'E_rs', 'K2', 'Vu_sv0',
#                'GV_dead', 'C_O2_param1', 'alpha2', 'Vu_ev0', 'Vu_jp', 'P_n', 'rise_time_ven',
#                'KcCO2', 'Fi_O2', 'Wb_sh', 'C_pv', 'Kv_tr', 'kes', 'fes_o', 'MO2_bp', 'fab_o',
#                'theta_v', 'GT_s', 'VA_rest', 'G_ap', 'Wp_v', 'beta2', 'fev_inf', 'k_ab', 'C_pp',
#                'fev_o', 'kev', 'T0', 'f_acCO2_n', 'GV_sv', 'Kp_tr', 'R_bpn', 'KE_rv', 'k_ac',
#                'KE_lv', 'theta_tr_max', 'Wc_v']

# subset_vars = ['V_tot', 'Vu_sv0', 'E_rs', 'R_rs', 'GV_sv', 'GV_dead', 'V0_dead', 'theta_v',
#  'C_O2_param1', 'Wp_v', 'G_ap', 'C_pv', 'VA_rest', 'Wc_v', 'Wb_sh', 'AT', 'Vu_jp',
#  'rise_time_ven', 'k_ab', 'Vu_ev0', 'k_ac', 'f_acCO2_n', 'Io_sh', 'fab_o',
#  'Kv_tr', 'Yv_max', 'tauMR', 'kev', 'PaO2_ac_n', 'P_n_max', 'fev_o', 'Fi_O2',
#  'C_pp', 'fes_o', 'GT_v', 'P_n', 'C_sv', 'KcCO2', 'fev_inf', 'GT_s', 'PaCO2_n',
#  'C2', 'MO2_bp', 'T0', 'Ysh_max', 'f_ac_max', 'Tc', 'Kp_tr', 'f_ab_max',
#  'fall_time_ven', 'theta_tr_max', 'R_po', 'a2', 'Kv_mi', 'KE_lv', 'kes', 'Io_sv',
#  'Kp_mi', 'GR_amp', 'Io_met', 'R_pv', 'KE_rv', 'f_ac_min', 'K2', 'Cvb_O2_n',
#  'phi_max', 'f_ab_min', 'Vu_bv', 'R_bpn', 'theta_mi_max', 'scale_param4',
#  'kcc_sh', 'Rvc_n']

# # Minute Ventilation: 7 parameters contribute 90% sensitivity
# subset_vars = ['R_rs', 'PaCO2_n', 'E_rs', 'C2', 'V0_dead', 'GV_dead', 'V_tot']
# # exercise
# subset_vars = ['R_rs', 'E_rs', 'GV_dead', 'V0_dead', 'PaCO2_n', 'VA_rest', 'KcCO2', 'V_tot',
#  'C_O2_param1', 'C2', 'MO2_bp', 'KcMRV']


subset_vars = [name for name in sp["names"] if name in subset_vars]

# Get all indices corresponding to subset_vars
# subset_idx = []
subset_idx = [sp['names'].index(var) for var in subset_vars if var in sp['names']]
subset_idx = np.array(subset_idx) # include full index range
X_subset = X_all[:, subset_idx]
subset_bounds = [sp['bounds'][i] for i in subset_idx]


sp_subset = ProblemSpec({
    'names': subset_vars,
    'bounds': subset_bounds
})

param_ranges: dict[str, tuple[float, float]] = {
    str(name): (float(b[0]), float(b[1]))
    if str(name) in subset_vars
    else (np.mean([float(b[0]), float(b[1])]), np.mean([float(b[0]), float(b[1])]))
    for name, b in zip(sp_subset["names"], sp_subset["bounds"])
}

Parameters1 = dict(zip(sp['names'], X_all[0,:]))
print(Parameters1)


output_names = [
    "Heart Rate", "Systolic Pressure", "Diastolic Pressure", "EDV", "ESV",
    "Max RV Volume", "Min RV Volume", "Max RV Pressure", "Min RV Pressure",
    "Min RA Volume", "Max RA Volume", "Min RA Pressure A descent", "Max RA Pressure Atrial contraction",
    "Max RA Pressure Tricuspid Opening", "Min RA Pressure V descent",
    "Min LA Volume", "Max LA Volume", "Min LA Pressure A descent", "Max LA Pressure Atrial contraction",
    "Max LA Pressure Tricuspid Opening", "Min LA Pressure V descent",
    "LA EDV", "RA EDV", "LV Pressure Deriv", "RV Pressure Deriv", "Tidal Volume", "Minute Ventilation",
    "Cardiac Output", "PaO2", "PaCO2", "Percentage Volume Change"]

Simulator = Cardiopulmonary(param_ranges=param_ranges, output_names=output_names)

hmw = HistoryMatchingWorkflow(
    simulator=Simulator,
    result=GaussianProcess_final,
    observations=observation,
    # optional parameters
    threshold=3.0,
    random_seed=42,
    # train_x=X,
    # train_y=Result,
    calibration_params=subset_vars,
)



N = X_subset.shape[0] // 2
D = len(subset_idx)

print(f"N = {N}")
print(f"D = {D}")


if calc_second_order:
    sample_size = N * (2 * D + 2)
else:
    sample_size = (D + 2) * N

skip_values = 0

base_sequence = np.zeros((N + skip_values, 2 * D),dtype=float)
base_sequence[:,:D] = X_subset[:N,:]
base_sequence[:,D:] = X_subset[N:(N*2),:]

saltelli_sequence = np.zeros([sample_size, D])

index = 0
for i in range(N):
    # Copy matrix "A"
    saltelli_sequence[index, :] = base_sequence[i, :D]
    index += 1

    # 2. Cross-sample hybrids (A with one column from B)
    for k in range(D):
        saltelli_sequence[index, :] = base_sequence[i, :D]
        saltelli_sequence[index, k] = base_sequence[i, k + D]
        index += 1

    # Copy matrix "B"
    saltelli_sequence[index, :] = base_sequence[i, D:]
    index += 1

    # Cross-sample elements of "A" into "B"
    # Only needed if you're doing second-order indices (true by default)
    if calc_second_order:
        for k in range(D):
            # Start with all columns from B
            saltelli_sequence[index, :] = base_sequence[i, D:]
            # Replace the k-th column with A
            saltelli_sequence[index, k] = base_sequence[i, k]
            index += 1

X = torch.from_numpy(saltelli_sequence.astype(np.float32))

Result_tensor, Var_tensor = GaussianProcess_final.model.predict_mean_and_variance(X)
Result = Result_tensor.detach().cpu().numpy()[:,0]
Var = Var_tensor.detach().cpu().numpy()[:,0]

Implaus = hmw.calculate_implausibility(Result_tensor, Var_tensor)
Implaus = Implaus.detach().cpu().numpy()[:,0]



## added code for removing entire A/B if even a single permutation is outside of the implausibility of 3
block_length = 2 * D + 2 if calc_second_order else D + 2
blocks = Implaus.reshape(N, block_length)
valid_mask = np.all(blocks <= 3.0, axis=1)  # True if all implausibilities ≤ 3.0
valid_indices = np.where(valid_mask)[0]

# Create a mask over all rows
row_mask = np.repeat(valid_mask, block_length)

# Filter everything in one go
filtered_saltelli = saltelli_sequence[row_mask]
filtered_Implaus = Implaus[row_mask]
filtered_Result = Result[row_mask]

index = 0
for i in valid_indices:
    start = i * block_length
    end = start + block_length
    filtered_saltelli[index:index + block_length, :] = saltelli_sequence[start:end, :]
    filtered_Implaus[index:index + block_length] = Implaus[start:end]
    filtered_Result[index:index + block_length] = Result[start:end]
    index += block_length

print(f"Number of base A/B blocks remaining: {len(valid_indices)}")
print(f"Number of base A/B blocks originally: {N}")


# Just HR plot
fig, ax1 = plt.subplots()
sns.kdeplot(filtered_Result, fill=True)

ax1.set_title(f"Filtered {Variable}")
ax1.set_xlabel("Value")
ax1.set_ylabel("Density")
plt.tight_layout()
plt.show()

X_scaled = scale_samples(filtered_saltelli, sp_subset)

ST = np.zeros((0, D), dtype=float)
S1 = np.zeros((0, D), dtype=float)

ST_std = np.zeros((0, D), dtype=float)
S1_std = np.zeros((0, D), dtype=float)

S = analyze_NIMP(sp_subset, filtered_Result.copy(), calc_second_order=calc_second_order, print_to_console=True)

T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S)

ST = np.vstack((ST, T_Si["ST"].reshape(1, -1)))
S1 = np.vstack((S1, first_Si["S1"].reshape(1, -1)))

conf_level = 0.95
z = norm.ppf(0.5 + conf_level / 2)

ST_std = np.vstack((ST_std, T_Si["ST_conf"].reshape(1, -1) / z))
S1_std = np.vstack((S1_std, first_Si["S1_conf"].reshape(1, -1) / z))


# --- Convert to DataFrame for plotting ---
param_names = sp_subset['names']  # assuming this exists
total = pd.DataFrame({
    "Parameter": param_names,
    "ST": ST.flatten(),
    "ST_std": ST_std.flatten(),
    "S1": S1.flatten(),
    "S1_std": S1_std.flatten()
}).set_index("Parameter")

# --- Sort by Total-order sensitivity ---
ranked = total.sort_values("ST", ascending=False)

ranked.to_csv(f"Plot_abstract/{Variable}_sensitivities.csv", index=True)

# --- Bar plot ---
fig, ax = plt.subplots(figsize=(6, 12))
ranked["ST"].plot(kind="barh", xerr=ranked["ST_std"], ax=ax, color="skyblue", edgecolor="k")
ax.invert_yaxis()
# ax.set_xscale("log")
ax.set_title(f"{Variable} Sobol Total-Order Sensitivities (Ranked)", fontsize=14)
ax.set_xlabel("Total-order index (ST)")

# Annotate each bar with rank
for i, (name, value) in enumerate(zip(ranked.index, ranked["ST"])):
    ax.text(value * 1.05, i, f"#{i+1}", va="center", ha="left", fontsize=9, color="blue")

plt.tight_layout()
plt.show()


ranked = total.sort_values("S1", ascending=False)

# --- Bar plot ---
fig, ax = plt.subplots(figsize=(6, 12))
ranked["S1"].plot(kind="barh", xerr=ranked["S1_std"], ax=ax, color="skyblue", edgecolor="k")
ax.invert_yaxis()
# ax.set_xscale("log")
ax.set_title("Sobol First-Order Sensitivities (Ranked)", fontsize=14)
ax.set_xlabel("First-order index (S1)")

# Annotate each bar with rank
for i, (name, value) in enumerate(zip(ranked.index, ranked["S1"])):
    ax.text(value * 1.05, i, f"#{i+1}", va="center", ha="left", fontsize=9, color="blue")

plt.tight_layout()
plt.show()


if calc_second_order:
    for j in range(D):
        for k in range(j + 1, D):
            print("%s %s %f %f" % (sp_subset["names"][j], sp_subset["names"][k],
                                   S['S2'][j, k], S['S2_conf'][j, k]))

    S2_list = second_Si["S2"]
    S2_conf_list = second_Si["S2_conf"] / z

    param_pairs = [(sp_subset["names"][i], sp_subset["names"][j]) for i in range(D) for j in range(i + 1, D)]


    S2_df = pd.DataFrame({
        "Parameter_pair": [" & ".join(pair) for pair in param_pairs],
        "S2": np.array(S2_list).flatten(),
        "S2_std": np.array(S2_conf_list).flatten()
    }).sort_values("S2", ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 16))
    S2_df.plot(
        kind="barh",
        x="Parameter_pair",
        y="S2",
        xerr="S2_std",
        ax=ax,
        color="skyblue",
        edgecolor="k"
    )
    ax.invert_yaxis()
    ax.set_title(f"{Variable} Sobol Second-Order Sensitivities", fontsize=14)
    ax.set_xlabel("Second-order index (S2)")
    plt.tight_layout()
    plt.show()