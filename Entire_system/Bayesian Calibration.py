import os

import pandas as pd
import torch
import warnings
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pyro
import arviz as az

from getdist.arviz_wrapper import arviz_to_mcsamples
from getdist import plots
from SALib import ProblemSpec
from autoemulate.simulations.epidemic import Epidemic
from autoemulate.core.compare import AutoEmulate
from autoemulate.calibration.bayes import BayesianCalibration
from autoemulate.emulators import GaussianProcess
from autoemulate.data.utils import set_random_seed
from autoemulate.calibration.history_matching import HistoryMatching, HistoryMatchingWorkflow
from sklearn.model_selection import KFold
from AutoEmulate_Simulator import Cardiopulmonary

# ----------------------------
# SETTINGS
# ----------------------------
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

random_seed = 42
set_random_seed(random_seed)
pyro.set_rng_seed(random_seed)

# ----------------------------
# PROBLEM SPECIFICATION
# ----------------------------
lower, upper = 0.5, 1.5

sp = ProblemSpec({
        'outputs': ["HR"],

        'names': [
        "beta2", "C2", "K2", "a2", "alpha2", "dc", "KCCO2",
        # "MRBCO2",
        "GV_dead",
        # "Kbg",
        "KcCO2", "KcMRV", "KpCO2", "KpO2", "V0_dead", "VA_rest", "Pmax",
        "Pmax_dot", "E_rs", "R_rs",
        # cardio
        "C_jp", "C_sa", "L_sa", "R_sa", "C_amv", "C_bv",
        "C_ev", "C_hv", "C_rmv", "C_sv", "kr_am", "P_0", "R_amv_n", "R_bv_n",
        "R_ev_n", "R_hv_n", "R_rmv_n", "R_sv_n", "D1", "K1_vc", "Kr_vc", "Rvc_n",
        "C_pa", "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv", "Emax_la", "P0_la", "Emax_ra",
        "P0_ra", "KE_la", "KE_ra", "P0_lv", "P0_rv", "g_abd", "g_thor", "P_abdmax_n", "P_abdmin_n",
        "P_thormax_n", "P_thormin_n",
        "VT_n", "A_im", "Tc", "T_im", "s",
        # cardio control
        "fab_o", "fes_o", "fes_inf", "fes_max", "fev_o", "fev_inf",
        "kes", "kev", "Io_sh", "Io_sp", "Io_sv", "Io_v", "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v", "Ysh_max",
        "Ysh_min", "Ysp_max", "Ysp_min",
        "Ysv_max", "Ysv_min", "Yv_max", "Yv_min", "theta_v", "Wb_sh", "Wb_sp", "Wb_sv", "Wc_sh", "Wc_sp",
        "Wc_sv", "Wc_v", "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
        "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
        "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "AT", "g_ccsh", "g_ccsp",
        "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv", "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn", "theta_spn",
        "theta_svn", "x_sh", "x_sp", "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min", "k_ab", "P_n", "P_n_max",
        "f_acCO2_n", "f_ac_max",
        "f_ac_min", "k_ac", "K_H", "PaO2_ac_n", "G_ap", "GT_s", "GT_v", "T0", "A", "B",
        "C", "D", "Cvb_O2_n", "gb_O2", "MO2_bp", "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2", "grm_O2",
        "Kh_CO2", "Krm_CO2", "MO2_hpn", "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n", "gam_O2", "gM", "Io_met", "kmet",
        "MO2_ampn", "phi_max", "phi_min",
        # added params
        "Kp_ao", "Kf_ao", "Kb_ao", "Kv_ao", "theta_ao_max", "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max",
        "Kp_po",
        "Kf_po", "Kb_po", "Kv_po", "theta_po_max", "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max", "alpha_O2",
        "R_po", "R_mi", "R_tr", "R_ao", "C_O2_param1", "C_O2_param2", "C_O2_param3", "PAMO2_nominal",
        "Vu_sa", "V_tot", "Vu_bv", "Vu_hv", "Vu_jp", "Vu_vc",
        "Vvc_max", "Vvc_min", "Vu_pa", "Vu_pp", "Vu_pv", "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "tau_Emax_lv",
        "tau_Emax_rv", "tau_Ramp", "tau_Rep", "tau_Rrmp", "tau_Rsp", "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv",
        "Vu_amv0", "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc", "tau_isc", "tau_p", "tau_z", "tau_ac", "tau_ap",
        "tau_Ts", "tau_Tv", "tau_CO2", "tau_O2", "tau_w", "tau_M", "tau_met", "DEmax_lv", "DEmax_rv", "DR_amp",
        "DR_ep", "DR_rmp", "DR_sp", "DV_amv", "DV_ev", "DV_rmv", "DV_sv", "DT_s", "DT_v", "Dmet", "Fi_CO2",
        "Fi_O2", "Ta", "KE_lv", "KE_rv", "T1", "T2", "VL_CO2", "VL_O2", "KCSFCO2", "VB", "tauMR", "VTCO2", "VTO2",
        "tau_MRV",
        "scale_param1", "scale_param2", "scale_param3", "scale_param4",
        "scale_param5", "scale_param6", "scale_param7", "scale_param8",
        "shift_param1", "shift_param2", "shift_param3", "shift_param4",
        "Pa_O2_lower", "rise_time_atr", "fall_time_atr", "rise_time_ven",
        "fall_time_ven", "ahead1", "theta_min", "delta_P"
        ],

    'bounds': [
        # change
# HR, EDP, ESP, Max RA pressure has
# [0.03255 * lower, 0.03255 * upper], [87 * 0.9, 87 * 1.1],
# [194.4 * 0.9, 194.4 * 1.1], [1.819 * 0.9, 1.819 * 1.1],
# [0.05591 * lower, 0.05591 * upper], [0.015 * lower, 0.015 * upper],

# Max RV pressure, EDV, PaO2, Minute Vent has
# [0.03255 * 0.9, 0.03255 * 1.1], [87 * 0.9, 87 * 1.1],
# [194.4 * 0.9, 194.4 * 1.1], [1.819 * 0.9, 1.819 * 1.1],
# [0.05591 * 0.9, 0.05591 * 1.1], [0.015 * lower, 0.015 * upper],
        # gas
        [0.03255 * lower, 0.03255 * upper], [87 * 0.9, 87 * 1.1],
        [194.4 * 0.9, 194.4 * 1.1], [1.819 * 0.9, 1.819 * 1.1],
        [0.05591 * lower, 0.05591 * upper], [0.015 * lower, 0.015 * upper],
        [346000 * lower, 346000 * upper],
        # [0.0009 * lower, 0.0009 * upper],
        # resp control
        [0.1698 * lower, 0.1698 * upper],
        # [17.4 * lower, 17.4 * upper],
        [0.2332 * lower, 0.2332 * upper],
        [1 * lower, 1 * upper], [0.2025 * lower, 0.2025 * upper], [4.72e-09 * lower, 4.72e-09 * upper],
        [0.1587 * lower, 0.1587 * upper], [0.067 * lower, 0.067 * upper], [100 * lower, 100 * upper],
        [1000 * lower, 1000 * upper], [21.9 * lower, 21.9 * upper], [3.02 * lower, 3.02 * upper],
        # cardio
        [3.72 * lower, 3.72 * upper],
        [0.28 * lower, 0.28 * upper], [0.00022 * lower, 0.00022 * upper], [0.06 * lower, 0.06 * upper],
        [4.4 * lower, 4.4 * upper],
        [5.71 * lower, 5.71 * upper], [10 * lower, 10 * upper],
        [1.57 * lower, 1.57 * upper],
        [3.28 * lower, 3.28 * upper], [31.11 * lower, 31.11 * upper], [24.17 * lower, 24.17 * upper],
        [3.93 * lower, 3.93 * upper],
        [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper],
        [0.224 * lower, 0.224 * upper], [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper],
        [0.3855 * lower, 0.3855 * upper], [0.15 * lower, 0.15 * upper],
        [0.001 * lower, 0.001 * upper], [0.05 * lower, 0.05 * upper],
        [0.76 * lower, 0.76 * upper], [15.8 * lower, 15.8 * upper], [25.37 * lower, 25.37 * upper],
        [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
        [0.1 * lower, 0.1 * upper], [0.35 * lower, 0.35 * upper], [0.55 * lower, 0.55 * upper],
        [0.35 * lower, 0.35 * upper], [0.55 * lower, 0.55 * upper], [0.05 * lower, 0.05 * upper],
        [0.05 * lower, 0.05 * upper], [1.5 * lower, 1.5 * upper],
        [1.5 * lower, 1.5 * upper], [3.39 * lower, 3.39 * upper], [6.8 * lower, 6.8 * upper],
        [-1 * upper, -1 * lower], [-2.5 * upper, -2.5 * lower],
        [-2 * upper, -2 * lower],
        [-6 * upper, -6 * lower],
        [0.73 * lower, 0.73 * upper], [30 * lower, 30 * upper],
        [0.7 * lower, 0.7 * upper], [1.1 * lower, 1.1 * upper], [0.04 * lower, 0.04 * upper],
        # cardio control
        [25 * lower, 25 * upper], [16.11 * lower, 16.11 * upper], [2.1 * lower, 2.1 * upper],
        [80 * lower, 80 * upper], [3.2 * lower, 3.2 * upper], [6.3 * lower, 6.3 * upper],
        [0.0675 * lower, 0.0675 * upper], [7.06 * lower, 7.06 * upper], [0.658 * lower, 0.658 * upper],
        [0.65 * lower, 0.65 * upper], [0.45 * lower, 0.45 * upper],
        [0.22 * lower, 0.22 * upper], [0.114 * lower, 0.114 * upper],
        [0.13 * lower, 0.13 * upper], [0.09 * lower, 0.09 * upper], [0.0162 * lower, 0.0162 * upper],
        [20 * lower, 20 * upper], [-0.0283 * upper, -0.0283 * lower], [5.5 * lower, 5.5 * upper],
        [-0.037 * upper, -0.037 * lower], [64.9 * lower, 64.9 * upper], [-0.437 * upper, -0.437 * lower],
        [1.9 * lower, 1.9 * upper], [-0.0008 * upper, -0.0008 * lower], [-0.68 * upper, -0.68 * lower],
        [-1.75 * upper, -1.75 * lower], [-1.1375 * upper, -1.1375 * lower], [-1.1375 * upper, -1.1375 * lower],
        [1 * lower, 1 * upper], [1.716 * lower, 1.716 * upper], [1.716 * lower, 1.716 * upper],
        [0.2 * lower, 0.2 * upper], [-0.2 * upper, -0.2 * lower], [-0.3997 * upper, -0.3997 * lower],
        [-0.3997 * upper, -0.3997 * lower], [-0.103 * upper, -0.103 * lower], [0.4 * lower, 0.4 * upper],
        [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper],
        [1.4 * lower, 1.4 * upper], [0.7 * lower, 0.7 * upper], [2.66 * lower, 2.66 * upper],
        [0.475 * lower, 0.475 * upper], [0.282 * lower, 0.282 * upper], [4.47 * lower, 4.47 * upper],
        [1.94 * lower, 1.94 * upper], [2.47 * lower, 2.47 * upper], [0.695 * lower, 0.695 * upper],
        [-28.29 * upper, -28.29 * lower], [-74.21 * upper, -74.21 * lower], [-28.29 * upper, -28.29 * lower],
        [-265.4 * upper, -265.4 * lower], [3.51 * lower, 3.51 * upper], [1.655 * lower, 1.655 * upper],
        [5.27 * lower, 5.27 * upper], [2.49 * lower, 2.49 * upper], [(1 / 60) * lower, (1 / 60) * upper],
        [1 * lower, 1 * upper], [1.5 * lower, 1.5 * upper], [0.2 * lower, 0.2 * upper],
        [6 * lower, 6 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
        [45 * lower, 45 * upper], [30 * lower, 30 * upper], [30 * lower, 30 * upper],
        [3.6 * lower, 3.6 * upper], [13.32 * lower, 13.32 * upper], [13.32 * lower, 13.32 * upper],
        [53 * lower, 53 * upper], [6 * lower, 6 * upper], [6 * lower, 6 * upper],
        [40 * 0.9, 40 * 1.1], [47.78 * lower, 47.78 * upper], [2.52 * lower, 2.52 * upper],
        [11.76 * lower, 11.76 * upper], [92 * lower, 92 * upper], [112 * lower, 112 * upper],
        [1.4 * lower, 1.4 * upper],
        [12.3 * lower, 12.3 * upper], [0.835 * lower, 0.835 * upper], [29.27 * lower, 29.27 * upper],
        [3 * lower, 3 * upper], [45 * lower, 45 * upper], [11.76 * lower, 11.76 * upper],
        [-0.13 * upper, -0.13 * lower], [0.09 * lower, 0.09 * upper], [0.58 * lower, 0.58 * upper],
        [20.9 * lower, 20.9 * upper], [92.8 * lower, 92.8 * upper], [10570 * lower, 10570 * upper],
        [-5.251 * upper, -5.251 * lower], [0.14 * lower, 0.14 * upper], [10 * lower, 10 * upper],
        [0.925 * lower, 0.925 * upper], [6.57 * lower, 6.57 * upper], [0.11 * lower, 0.11 * upper],
        [0.155 * lower, 0.155 * upper], [35 * lower, 35 * upper], [30 * lower, 30 * upper],
        [11.11 * lower, 11.11 * upper], [142.8 * lower, 142.8 * upper], [0.4 * lower, 0.4 * upper],
        [0.86 * lower, 0.86 * upper], [19.71 * lower, 19.71 * upper], [12660 * lower, 12660 * upper],
        [0.1555 * lower, 0.1555 * upper], [30 * lower, 30 * upper], [40 * lower, 40 * upper],
        [0.4266 * lower, 0.4266 * upper],
        [0.18 * lower, 0.18 * upper], [0.516 * lower, 0.516 * upper], [20 * lower, 20 * upper],
        [-1.87 * upper, -1.87 * lower],
        # added params
        [1000 * lower, 1000 * upper], [5000 * lower, 5000 * upper], [2 * lower, 2 * upper],
        [5 * lower, 5 * upper], [1.309 * lower, 1.309 * upper], [100 * lower, 100 * upper],
        [500 * lower, 500 * upper], [2 * lower, 2 * upper], [7 * lower, 7 * upper],
        [1.309 * lower, 1.309 * upper], [3000 * lower, 3000 * upper], [2000 * lower, 2000 * upper],
        [5 * lower, 5 * upper], [10 * lower, 10 * upper], [1.309 * lower, 1.309 * upper],
        [100 * lower, 100 * upper], [500 * lower, 500 * upper], [2 * lower, 2 * upper],
        [7 * lower, 7 * upper], [1.309 * lower, 1.309 * upper], [0.0000317 * lower, 0.0000317 * upper],
        [350 * lower, 350 * upper], [350 * lower, 350 * upper], [350 * lower, 350 * upper],
        [350 * lower, 350 * upper], [0.00134 * lower, 0.00134 * upper],
        [2.6 * lower, 2.6 * upper], [3.03e-5 * lower, 3.03e-5 * upper], [104 * lower, 104 * upper],
        [1 * lower, 1 * upper], [5027.6 * 0.8, 5027.6 * 1.2], [279.49 * lower, 279.49 * upper],
        [93.16 * lower, 93.16 * upper],
        [579.76 * lower, 579.76 * upper], [123 * lower, 123 * upper], [350 * lower, 350 * upper],
        [50 * lower, 50 * upper], [1 * lower, 1 * upper], [116.6775 * lower, 116.6775 * upper],
        [114 * lower, 114 * upper], [4 * lower, 4 * upper], [15.908 * lower, 15.908 * upper],
        [4 * lower, 4 * upper], [38.703 * lower, 38.703 * upper], [8 * lower, 8 * upper],
        [8 * lower, 8 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [20 * lower, 20 * upper],
        [20 * lower, 20 * upper], [20 * lower, 20 * upper], [20 * lower, 20 * upper],
        [286.4 * lower, 286.4 * upper], [607.8 * lower, 607.8 * upper], [190.95 * lower, 190.95 * upper],
        [1361.6 * lower, 1361.6 * upper], [20 * lower, 20 * upper], [30 * lower, 30 * upper],
        [2.076 * lower, 2.076 * upper], [0.8 * lower, 0.8 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [1.5 * lower, 1.5 * upper],
        [20 * lower, 20 * upper], [10 * lower, 10 * upper], [5 * lower, 5 * upper],
        [40 * lower, 40 * upper], [10 * lower, 10 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [5 * lower, 5 * upper],
        [5 * lower, 5 * upper], [5 * lower, 5 * upper], [5 * lower, 5 * upper],
        [2 * lower, 2 * upper], [0.2 * lower, 0.2 * upper], [4 * lower, 4 * upper],
        [0.0421 * lower, 0.0421 * upper], [21.0379 * lower, 21.0379 * upper], [5 * lower, 5 * upper],
        [0.014 * lower, 0.014 * upper], [0.011 * lower, 0.011 * upper],
        [1 * lower, 1 * upper], [2 * lower, 2 * upper], [3 * lower, 3 * upper],
        [2.5 * lower, 2.5 * upper], [20 * lower, 20 * upper], [0.9 * lower, 0.9 * upper],
        [50 * lower, 50 * upper], [0.25 * lower, 0.25 * upper], [0.25 * lower, 0.25 * upper],
        [50 * lower, 50 * upper],
        # further added params
        [4.9 * lower, 4.9 * upper], [1.5 * lower, 1.5 * upper], [0.3 * lower, 0.3 * upper],
        [26.6 * lower, 26.6 * upper], [0.5 * lower, 0.5 * upper], [1.2 * lower, 1.2 * upper],
        [30 * lower, 30 * upper], [1.6 * lower, 1.6 * upper], [4 * lower, 4 * upper],
        [0.3 * lower, 0.3 * upper], [4 * lower, 4 * upper], [0.3 * lower, 0.3 * upper],
        [80 * lower, 80 * upper], [0.05 * lower, 0.05 * upper], [0.1 * lower, 0.1 * upper],
        [0.15 * lower, 0.15 * upper], [0.3 * lower, 0.3 * upper], [0.85 * 0.9, 0.85 * 1.1],
        [0.0872665 * lower, 0.0872665 * upper], [0.3 * lower, 0.3 * upper]]
})

# subset_vars = ['k_ac', 'Wp_sv', 'ahead1', 'theta_min', 'delta_P', 'G_ap', 'Cvh_O2_n', 'T_im', 'K1_vc',
#                'theta_mi_max', 'P0_rv', 'GT_s', 'theta_svn', 'Emax_lv0', 'f_ac_min',
#                'kmet', 'R_sa', 'R_bpn', 'Io_sv', 'phi_max', 'R_po', 'f_acCO2_n', 'Kv_tr', 'Emax_rv0', 'V_tot',
#                'kes', 'Io_met', 'Cvam_O2_n', 'fev_inf', 'theta_spn', 'theta_tr_max', 'Wb_sh',
#                'C_pp', 'Vu_hv', 'g_ccsp', 'R_mi', 'f_ab_max', 'Wb_sv', 'Tc', 'GEmax_rv', 'GEmax_lv',
#                'Vu_bv', 'KE_lv', 'Wc_sp', 'scale_param2', 'KE_ra', 'GR_ep', 'Vu_rv', 'fes_min', 'Ysv_min',
#                'k_ab', 'R_pv', 'grm_O2', 'KE_la', 'fes_o', 'Vu_vc', 'GR_sp',
#                'Cvrm_O2_n', 'C_jp', 'Wc_v', 'C_pv', 'g_ccsh', 'C_sv', 'MO2_rmp', 'rise_time_ven',
#                'Ysv_max', 'Vu_amv0', 'KE_rv', 'Vu_lv', 'Vu_ev0', 'GT_v', 'R_amp0', 'D', 'gb_O2',
#                'f_ac_max', 'theta_v', 'theta_shn', 'kcc_sv', 'kev', 'fes_inf', 'MO2_ampn', 'Vu_rmv0', 'Wb_sp',
#                'Vu_jp', 'Cvb_O2_n', 'Kv_po', 'Wp_v', 'Rvc_n', 'R_rmp0', 'R_sp0',
#                'kcc_sh', 'Kp_tr', 'GV_sv', 'T0', 'fev_o', 'R_tr', 'theta_po_max', 'f_ab_min',
#                'R_hv_n', 'R_pa', 'P_thormin_n', 'Vu_sv0', 'fab_o', 'phi_min', 'fall_time_ven', 'Wp_sh', 'Kp_po',
#                'P0_lv', 'R_ep0', 'Vu_pv', 'C_ev', 'MO2_bp', 'Wc_sh', 'P_n', 'Vu_pp', 'R_pp',
#                # comment out to remove the below to just focus on the cardiovascular variables
#                'beta2', 'C2', 'K2', 'a2', 'alpha2', 'GV_dead', 'KcCO2', 'KpCO2', 'Fi_O2', 'V0_dead',
#                'VA_rest', 'E_rs', 'R_rs', 'PaCO2_n', 'C_O2_param1', 'C_O2_param2', 'PaO2_ac_n',
#                'scale_param3', 'scale_param4', 'K_H'
#                ]

# change
#     # HR: 17 parameters contribute 90 % sensitivity
subset_vars = ['T0', 'V_tot', 'P_n', 'fev_o', 'GT_v', 'GT_s', 'C2', 'C_O2_param1', 'Fi_O2',
 'Vu_sv0', 'fes_o', 'fab_o', 'kes', 'Wb_sh', 'K2', 'k_ab', 'f_acCO2_n']

# Max RV Pressure: 46 parameters contribute 90% sensitivity
# subset_vars = ['V_tot', 'PaCO2_n', 'C2', 'R_rs', 'a2', 'V0_dead', 'E_rs', 'K2', 'Vu_sv0',
#                'GV_dead', 'C_O2_param1', 'alpha2', 'Vu_ev0', 'Vu_jp', 'P_n', 'rise_time_ven',
#                'KcCO2', 'Fi_O2', 'Wb_sh', 'C_pv', 'Kv_tr', 'kes', 'fes_o', 'MO2_bp', 'fab_o',
#                'theta_v', 'GT_s', 'VA_rest', 'G_ap', 'Wp_v', 'beta2', 'fev_inf', 'k_ab', 'C_pp',
#                'fev_o', 'kev', 'T0', 'f_acCO2_n', 'GV_sv', 'Kp_tr', 'R_bpn', 'KE_rv', 'k_ac',
#                'KE_lv', 'theta_tr_max', 'Wc_v']

# #   EDP: 83 parameters contribute 90% sensitivity
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

# # EDV: 12 parameters contribute 90% sensitivity
# subset_vars = ['V_tot', 'Vu_sv0', 'Emax_lv0', 'T0', 'Vu_ev0', 'Vu_jp', 'fall_time_ven', 'C_pv',
#                 'C_O2_param1', 'Kv_tr', 'C_pp', 'KE_lv']

# #   Systolic Pressure: 28 parameters contribute 90% sensitivity
# subset_vars = ['V_tot', 'Vu_sv0', 'P_n', 'C2', 'PaCO2_n', 'kes', 'a2', 'V0_dead', 'fes_o', 'R_rs',
#                'E_rs', 'GV_dead', 'Vu_ev0', 'K2', 'Vu_jp', 'C_pv', 'fes_min', 'R_pv', 'R_sa',
#                'Fi_O2', 'Cvrm_O2_n', 'C_O2_param1', 'fab_o', 'rise_time_ven', 'fall_time_ven',
#                'GV_sv', 'C_pp', 'Kv_tr']

#    # Max RA Pressure: 89 parameters contribute 90 % sensitivity
# subset_vars = ['a2', 'Vu_sv0', 'MO2_bp', 'PaCO2_n', 'C2', 'G_ap', 'Wp_v', 'R_rs', 'kes',
#  'V0_dead', 'GT_v', 'GV_dead', 'theta_v', 'K2', 'PaO2_ac_n', 'beta2', 'V_tot',
#  'E_rs', 'Fi_O2', 'kev', 'fev_o', 'GV_sv', 'Wb_sh', 'T0', 'f_ab_max', 'fes_o',
#  'Wc_v', 'f_acCO2_n', 'P_n', 'alpha2', 'C_O2_param1', 'k_ab', 'GT_s',
#  'scale_param4', 'Cvb_O2_n', 'KcCO2', 'C_pv', 'fev_inf', 'Vu_ev0', 'Kv_mi',
#  'fes_min', 'Vu_jp', 'fab_o', 'f_ac_max', 'theta_mi_max', 'f_ab_min', 'KE_ra',
#  'C_pp', 'k_ac', 'theta_shn', 'Vu_bv', 'VA_rest', 'Cvrm_O2_n', 'f_ac_min',
#  'R_bpn', 'Io_met', 'Cvam_O2_n', 'grm_O2', 'KE_lv', 'fall_time_ven', 'Emax_ra',
#  'Kv_tr', 'C_sv', 'KE_rv', 'Wb_sp', 'kmet', 'P0_rv', 'theta_svn', 'g_ccsh',
#  'C_O2_param2', 'Wc_sh', 'Wp_sp', 'KpCO2', 'C_jp', 'Vu_rmv0', 'R_pv', 'Kp_mi',
#  'fes_inf', 'Io_sh', 'theta_tr_max', 'Vu_vc', 'Io_sp', 'R_sp0', 'kcc_sh', 'C_sa',
#  'C_bv', 'MO2_hpn', 'Vu_rv', 'Vu_pp']

# # PaO2: 2 parameters contribute 90% sensitivity
# subset_vars = ['Fi_O2', 'PaCO2_n']

# # Minute Ventilation: 7 parameters contribute 90% sensitivity
# subset_vars = ['R_rs', 'PaCO2_n', 'E_rs', 'C2', 'V0_dead', 'GV_dead', 'V_tot']

# MUST SORT SO ITS THE SAME ORDER
subset_vars = [name for name in sp["names"] if name in subset_vars]


# Convert to dictionary
param_ranges: dict[str, tuple[float, float]] = {
    str(name): (float(b[0]), float(b[1]))
    if str(name) in subset_vars
    else (np.mean([float(b[0]), float(b[1])]), np.mean([float(b[0]), float(b[1])]))
    for name, b in zip(sp["names"], sp["bounds"])
}

# output_names = [
#     "Heart Rate", "Systolic Pressure", "Diastolic Pressure", "EDV", "ESV",
#     "Max RV Volume", "Min RV Volume", "Max RV Pressure", "Min RV Pressure",
#     "Min RA Volume", "Max RA Volume", "Min RA Pressure", "Max RA Pressure",
#     "Min LA Volume", "Max LA Volume", "Min LA Pressure", "Max LA Pressure",
#     "LA ESV", "RA ESV", "LV Pressure Deriv", "RV Pressure Deriv"]
    # "Stroke Volume", "Ejection Fraction"]

output_names = ["Heart Rate"]

# ----------------------------
# LOAD SIMULATOR
# ----------------------------
Simulator = Cardiopulmonary(param_ranges=param_ranges, output_names=output_names)


# ----------------------------
# LOAD EMULATOR
# ----------------------------
# change
Variable = "HR"
# GaussianProcess_final = joblib.load("best_emulator/Max_RV_P_GaussianProcessMatern32_5000.joblib")
GaussianProcess_final= joblib.load(f"best_GaussianProcessMatern32/best_emulator/{Variable}_GaussianProcessMatern32_10000.joblib")
# ----------------------------
# OBSERVATION
# ----------------------------
# change
# observation = {"EDP": (70, 3)}
observation = {"HR": (1.1, 0.1)}
# observation = {"Max_RV_P": (25, 2)}
# observation = {"EDV": (163, 23)}
# observation = {"ESP": (105, 5)}
# observation = {"Max_RA_P": (7, 1)}
# observation = {"PaO2": (87.5, 4.2)}
# observation = {"Minute_vent": (6.5, 0.5)}

# ----------------------------
# BAYESIAN CALIBRATION
# ----------------------------
if __name__ == "__main__":
    # X_all = np.load('DGSM_filtered_LHCS_500000_X_sample_21_targets_rest.npy')
    # Result_all = np.load('DGSM_filtered_LHCS_500000_Result_21_targets_rest.npy')
    # mask = Result_all[:, 0] != 0
    # X = X_all[mask, :]
    # Result = Result_all[mask, :]
    #
    # nan_mask = ~np.isnan(Result).any(axis=1)  # True for rows without NaN
    # X = X[nan_mask, :]
    # Result = Result[nan_mask, :]
    #
    # # get the mean of the column
    # col_mean = Result.mean(axis=0)
    # col_std = Result.std(axis=0)
    # # 3 std to remove outliers
    # mask = np.all((Result >= (col_mean - 2*col_std)) & (Result <= (col_mean + 2*col_std)), axis=1)
    # X = X[mask, :]
    # Result = Result[mask]
    #
    # X = X[:1000,:]
    # Result = Result[:1000]
    #
    # X = torch.tensor(X, dtype=torch.float32)
    # Result = torch.tensor(Result[:,0], dtype=torch.float32).unsqueeze(1)

    hmw = HistoryMatchingWorkflow(
        simulator=Simulator,
        result=GaussianProcess_final,
        observations=observation,
        # optional parameters
        threshold=3.0,
        random_seed=random_seed,
        # train_x=X,
        # train_y=Result,
        calibration_params=subset_vars,
    )

    _ = hmw.run_waves(n_waves=1, n_simulations=20, n_test_samples=10000, refit_on_all_data=False, refit_emulator_on_last_wave=False)

    hmw.plot_wave((len(hmw.wave_results)-1), fname=f"{Variable}_10000_wave_{(len(hmw.wave_results)-1)}.png")
    print(len(hmw.wave_results)-1)
    # hmw.plot_wave(1, fname="save_RV_max_1.png")

    # Get the last wave results
    test_parameters, impl_scores = hmw.wave_results[-1]
    nroy_points = hmw.get_nroy(impl_scores, test_parameters)  # Implausibility < 3.0

    # Get exact min/max bounds for the parameters from the NROY points
    params_post_hm = hmw.generate_param_bounds(
        nroy_x=nroy_points,
        param_names=sp["names"],
        buffer_ratio=0.0
    )

    np.save ("NROY_Points_HR.npy", nroy_points)
    np.save("NROY_Params_HR.npy", params_post_hm)

    #
    # model_post_hm = hmw.emulator  # Use the emulator from history matching
    # parameter_range = {k: v for k, v in params_post_hm.items() if k in subset_vars}
    #
    # bc = BayesianCalibration(
    #     emulator=model_post_hm,
    #     parameter_range=parameter_range,
    #     observations={k: torch.tensor(v[0]) for k, v in observation.items()},
    #     # take account of the emulator uncertainty
    #     model_uncertainty=True,
    #     # specify observation noise as variance
    #     observation_noise={k: v[1] ** 2 for k, v in observation.items()}
    # )
    #
    # mcmc = bc.run_mcmc(warmup_steps=250, num_samples=1000, sampler='nuts')
    #
    # print(mcmc.summary())
    # summary_dict = mcmc.summary()
    #
    # df_summary = pd.DataFrame(summary_dict).T  # transpose to make parameters as rows
    # df_summary.to_csv(f"mcmc_summary_{Variable}.csv")
    #
    # with open(f"mcmc_summary_{Variable}.txt", "w") as f:
    #     f.write(df_summary.to_string())




















    # # Create BayesianCalibration object
    # bc = BayesianCalibration(
    #     GaussianProcess_final,
    #     param_ranges,
    #     observation,
    #     observation_noise=20.0,
    # )
    #
    # # Run MCMC
    # mcmc_emu = bc.run_mcmc(
    #     warmup_steps=250,
    #     num_samples=1000,
    #     num_chains=2,
    #     sampler="metropolis",
    # )
    #
    # # Summarize and plot
    # print(mcmc_emu.summary())
    # az_data = bc.to_arviz(mcmc_emu, posterior_predictive=True)
    # print(az_data)
    # import arviz as az
    #
    # _ = az.plot_trace(az_data, figsize=(20, 8))
    #
    # _ = az.plot_ppc(az_data, show=True)
