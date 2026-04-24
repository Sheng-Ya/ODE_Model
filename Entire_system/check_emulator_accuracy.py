import os
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. PATHS — adjust these if needed ──────────────────────────────
# Assumes you run from the Max_LA_Volume folder
Var = "Max_LA_Volume"
EMULATOR_PATH = f"C:/Users/vanes/Downloads/exercise_model/ODE_Exercise/Entire_system/Emulator_wave_1wave/{Var}/GaussianProcessMatern32_{Var}_best.joblib"

# X_train.pt and Y_train.pt were saved by pre_wave_train_emulators
# They're likely in the parent working directory where you launched the script
X_TRAIN_PATH = "LHCS_X_20.npy "  # adjust if needed
Y_TRAIN_PATH = "LHCS_Result_20.npy "  # adjust if needed

# Max_LA_Volume is output index 14 in the full output list
TARGET_COL = 14

# ── 2. BUILD parameter_idx ─────────────────────────────────────────
# Reproducing exactly what HistoryMatchingWorkflow.__init__ does:
# self.parameter_idx = [simulator.get_parameter_idx(p) for p in calibration_params]
#
# Since subset_vars is sorted by sp["names"] order, and get_parameter_idx
# returns the positional index in sp["names"], we reconstruct it here.

from SALib import ProblemSpec
lower = 0.8
upper = 1.2
sp = ProblemSpec({
    'names': [
        # gas
        "beta2", "C2", "K2", "a2",
        "alpha2", "KCCO2", "GV_dead",
        # resp control
        "KcCO2", "KcMRV", "KpCO2", "KpO2",
        "V0_dead", "VA_rest",
        "E_rs", "R_rs",
        # cardio
        "C_jp", "C_sa", "L_sa", "R_sa",
        "C_amv", "C_bv", "C_ev", "C_hv",
        "C_rmv", "C_sv", "kr_am", "P_0",
        "R_amv_n", "R_bv_n", "R_ev_n", "R_hv_n",
        "R_rmv_n", "R_sv_n", "K1_vc", "D1",
        "Vvc_min", "Kr_vc",
        "Rvc_n", "C_pa", "C_pp",
        "C_pv", "L_pa", "R_pa", "R_pp",
        "R_pv", "Emax_la", "P0_la", "Emax_ra",
        "P0_ra", "KE_la", "KE_ra", "P0_lv",
        "P0_rv",
        "s",
        # cardio control
        "fab_o", "fes_o", "fes_inf", "fes_max",
        "fev_o", "fev_inf", "kes", "kev",
        "Io_sh", "Io_sp", "Io_sv", "Io_v",
        "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v",
        "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min",
        "Ysv_max", "Ysv_min", "Yv_max", "Yv_min",
        "theta_v", "Wb_sh", "Wb_sp", "Wb_sv",
        "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v",
        "Wp_sp", "Wp_sv", "Wp_v",
        "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
        "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv",
        "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp",
        "GR_sp", "GV_amv", "GV_ev", "GV_rmv",
        "GV_sv", "R_amp0", "R_ep0", "R_rmp0",
        #
        "R_sp0", "g_ccsh", "g_ccsp",
        "kisc_sh", "kisc_sp", "kisc_sv",
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
        "PAMO2_nominal", "Vu_bv", "Vu_hv",
        "Vu_jp", "Vu_vc",
        "Vu_pp", "Vu_pv", "Vu_la", "Vu_lv",
        "Vu_ra", "Vu_rv",

        "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp",
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
        "scale_param1", "scale_param3", "scale_param4",
        "scale_param6", "Pa_O2_lower",
        "rise_time_atr", "rise_time_ven", "fall_time_ven", "ahead1",
        "theta_min", "r", "l", "V_nominal", "V_scale"
    ],

    'bounds': [
        # gas
        [0.03255 * lower, 0.03255 * upper], [87 * lower, 87 * upper], [194.4 * lower, 194.4 * upper],
        [1.819 * lower, 1.819 * upper],
        [0.05591 * lower, 0.05591 * upper], [346000 * lower, 346000 * upper], [0.1698 * lower, 0.1698 * upper],
        # resp control
        [0.2332 * lower, 0.2332 * upper], [1 * lower, 1 * upper], [0.2025 * lower, 0.2025 * upper],
        [4.72e-09 * lower, 4.72e-09 * upper],
        [0.1587 * lower, 0.1587 * upper], [0.0673 * lower, 0.0673 * upper],
        [21.9 * 0.8, 21.9 * 1.2], [3.02 * 0.8, 3.02 * 1.2],
        # cardio
        [3.72 * lower, 3.72 * upper], [0.28 * lower, 0.28 * upper], [0.00022 * lower, 0.00022 * upper],
        [0.06 * lower, 0.06 * upper],
        [9.4 * lower, 9.4 * upper], [10.71 * lower, 10.71 * upper], [20 * lower, 20 * upper],
        [3.57 * lower, 3.57 * upper],
        [6.28 * lower, 6.28 * upper], [61.11 * lower, 61.11 * upper], [24.17 * lower, 24.17 * upper],
        [10 * lower, 10 * upper],
        [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper],
        [0.224 * lower, 0.224 * upper],
        [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper], [0.15 * lower, 0.15 * upper],
        [0.3855 * lower, 0.3855 * upper],
        [50 * lower, 50 * upper], [10000 * lower, 10000 * upper],
        [0.025 * lower, 0.025 * upper], [0.76 * lower, 0.76 * upper], [5.8 * lower, 5.8 * upper],
        [25.37 * lower, 25.37 * upper], [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper],
        [0.0894 * lower, 0.0894 * upper],
        [0.0056 * lower, 0.0056 * upper], [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper],
        [0.45 * lower, 0.45 * upper],
        [0.45 * lower, 0.45 * upper], [0.05 * lower, 0.05 * upper], [0.05 * lower, 0.05 * upper],
        [1.5 * lower, 1.5 * upper],
        [1.5 * lower, 1.5 * upper],
        [0.04 * lower, 0.04 * upper],
        # cardio control
        [25 * lower, 25 * upper], [16.11 * lower, 16.11 * upper], [2.1 * lower, 2.1 * upper], [80 * lower, 80 * upper],
        [3.2 * lower, 3.2 * upper], [6.3 * lower, 6.3 * upper], [0.0675 * lower, 0.0675 * upper],
        [7.06 * lower, 7.06 * upper],
        [0.658 * lower, 0.658 * upper], [0.65 * lower, 0.65 * upper], [0.45 * lower, 0.45 * upper],
        [0.126 * lower, 0.126 * upper],
        [0.114 * lower, 0.114 * upper], [0.13 * lower, 0.13 * upper], [0.09 * lower, 0.09 * upper],
        [0.0162 * lower, 0.0162 * upper],
        [9 * lower, 9 * upper], [-0.0283 * upper, -0.0283 * lower], [5.5 * lower, 5.5 * upper],
        [-0.037 * upper, -0.037 * lower],
        [64.9 * lower, 64.9 * upper], [-0.437 * upper, -0.437 * lower], [1.9 * lower, 1.9 * upper],
        [-0.0008 * upper, -0.0008 * lower],
        [-0.68 * upper, -0.68 * lower], [-1.75 * upper, -1.75 * lower], [-1.1375 * upper, -1.1375 * lower],
        [-1.1375 * upper, -1.1375 * lower],
        [1 * lower, 1 * upper], [1.716 * lower, 1.716 * upper], [1.716 * lower, 1.716 * upper],
        [0.2 * lower, 0.2 * upper],
        [-0.3997 * upper, -0.3997 * lower], [-0.3997 * upper, -0.3997 * lower], [-0.103 * upper, -0.103 * lower],
        [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper],
        [2.392 * lower, 2.392 * upper], [1.412 * lower, 1.412 * upper], [2.66 * lower, 2.66 * upper],
        [0.475 * lower, 0.475 * upper],
        [0.282 * lower, 0.282 * upper], [2.47 * lower, 2.47 * upper], [1.94 * lower, 1.94 * upper],
        [2.47 * lower, 2.47 * upper],
        [0.695 * lower, 0.695 * upper], [-58.29 * upper, -58.29 * lower], [-74.21 * upper, -74.21 * lower],
        [-58.29 * upper, -58.29 * lower],
        [-265.4 * upper, -265.4 * lower], [3.51 * lower, 3.51 * upper], [1.655 * lower, 1.655 * upper],
        [5.27 * lower, 5.27 * upper],
        #
        [2.49 * lower, 2.49 * upper], [1 * lower, 1 * upper], [1.5 * lower, 1.5 * upper],
        [6 * lower, 6 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
        [45 * lower, 45 * upper], [30 * lower, 30 * upper], [30 * lower, 30 * upper], [3.6 * lower, 3.6 * upper],
        [13.32 * lower, 13.32 * upper], [13.32 * lower, 13.32 * upper], [53 * lower, 53 * upper],
        [6 * lower, 6 * upper],
        [6 * lower, 6 * upper], [40 * lower, 40 * upper], [47.78 * lower, 47.78 * upper], [2.52 * lower, 2.52 * upper],
        [11.76 * lower, 11.76 * upper], [92 * lower, 92 * 1.05], [112 * 0.9, 112 * upper], [1.4 * lower, 1.4 * upper],
        [12.3 * lower, 12.3 * upper], [0.835 * lower, 0.835 * upper], [29.27 * lower, 29.27 * upper],
        [3 * lower, 3 * upper],
        [45 * lower, 45 * upper], [11.76 * lower, 11.76 * upper], [-0.13 * upper, -0.13 * lower],
        [0.09 * lower, 0.09 * upper],
        [0.58 * lower, 0.58 * upper], [20.9 * lower, 20.9 * upper], [92.8 * lower, 92.8 * upper],
        [10570 * lower, 10570 * upper],
        [-5.251 * upper, -5.251 * lower], [0.14 * lower, 0.14 * upper], [10 * lower, 10 * upper],
        [0.925 * lower, 0.925 * upper],
        [6.57 * lower, 6.57 * upper], [0.11 * lower, 0.11 * upper], [0.155 * lower, 0.155 * upper],
        [35 * lower, 35 * upper],
        [30 * lower, 30 * upper], [11.11 * lower, 11.11 * upper], [142.8 * lower, 142.8 * upper],
        [0.4 * lower, 0.4 * upper],
        [0.86 * lower, 0.86 * upper], [19.71 * lower, 19.71 * upper], [12660 * lower, 12660 * upper],
        [0.1555 * lower, 0.1555 * upper],
        [30 * lower, 30 * upper], [40 * lower, 40 * upper], [0.4266 * lower, 0.4266 * upper],
        [0.18 * lower, 0.18 * upper],
        [0.516 * lower, 0.516 * upper], [20 * lower, 20 * upper], [-1.87 * upper, -1.87 * lower],
        # added params
        [1000 * lower, 1000 * upper], [5000 * lower, 5000 * upper], [2 * lower, 2 * upper], [7 * lower, 7 * upper],
        [1.309 * lower, 1.309 * upper],
        [1200 * lower, 1200 * upper], [200 * lower, 200 * upper], [2 * lower, 2 * upper], [3.5 * lower, 3.5 * upper],
        [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper], [2000 * lower, 2000 * upper], [2 * lower, 2 * upper], [7 * lower, 7 * upper],
        [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper], [200 * lower, 200 * upper], [2 * lower, 2 * upper], [3.5 * lower, 3.5 * upper],
        [1.309 * lower, 1.309 * upper],
        [0.0000317 * lower, 0.0000317 * upper], [350 * lower, 350 * upper], [400 * lower, 400 * upper],
        [400 * lower, 400 * upper],
        [350 * lower, 350 * upper], [0.00134 * lower, 0.00134 * upper], [2.6 * lower, 2.6 * upper],
        [3.03e-5 * lower, 3.03e-5 * upper],
        [104 * lower, 104 * upper], [279.49 * lower, 279.49 * upper], [93.16 * lower, 93.16 * upper],
        [579.76 * lower, 579.76 * upper], [123 * lower, 123 * upper],
        [116.68 * lower, 116.68 * upper], [114 * lower, 114 * upper], [33 * lower, 33 * upper],
        [15.908 * lower, 15.908 * upper],
        [30 * lower, 30 * upper], [38.703 * lower, 38.703 * upper],

        [8 * lower, 8 * upper], [8 * lower, 8 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [20 * lower, 20 * upper],
        [20 * lower, 20 * upper], [20 * lower, 20 * upper], [20 * lower, 20 * upper], [286.4 * lower, 286.4 * upper],
        [607.8 * lower, 607.8 * upper], [190.95 * lower, 190.95 * upper], [1361.6 * lower, 1361.6 * upper],
        [20 * lower, 20 * upper],
        [30 * lower, 30 * upper], [2.076 * lower, 2.076 * upper], [0.8 * lower, 0.8 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [1.5 * lower, 1.5 * upper], [20 * lower, 20 * upper],
        [10 * lower, 10 * upper], [5 * lower, 5 * upper], [40 * lower, 40 * upper], [10 * lower, 10 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [5 * lower, 5 * upper], [5 * lower, 5 * upper],
        [5 * lower, 5 * upper], [5 * lower, 5 * upper], [2 * lower, 2 * upper], [0.2 * lower, 0.2 * upper],
        [4 * lower, 4 * upper], [0.3 * lower, 0.3 * upper], [0.014 * lower, 0.014 * upper],
        [0.011 * lower, 0.011 * upper],
        [0.1 * lower, 0.1 * upper], [0.2 * lower, 0.2 * upper], [3 * lower, 3 * upper], [2.5 * lower, 2.5 * upper],
        [20 * lower, 20 * upper], [0.01 * lower, 0.01 * upper], [50 * lower, 50 * upper], [0.25 * lower, 0.25 * upper],
        [0.25 * lower, 0.25 * upper], [50 * lower, 50 * upper],

        # further added params
        [4.9 * lower, 4.9 * upper], [0.3 * lower, 0.3 * upper], [26.6 * lower, 26.6 * upper],
        [0.04 * lower, 0.04 * upper], [80 * lower, 80 * upper],
        [0.045 * lower, 0.045 * upper], [0.3 * lower, 0.3 * upper], [0.45 * 0.85, 0.45 * 1.15],
        [0.92 * 0.92, 0.92 * 1.08],
        [0.0873 * lower, 0.0873 * upper], [1.2 * 0.85, 1.2 * 1.15], [1.2 * 0.85, 1.2 * 1.15],
        [150 * lower, 150 * upper], [50 * lower, 50 * upper]]
})

# Exercise subset_vars (sorted by sp["names"] order, same as your script)
subset_vars_set = {
    'a2', 'ahead1', 'beta2', 'C2', 'C_jp', 'C_O2_param1', 'C_sv', 'Cvam_O2_n', 'E_rs', 'Emax_la',
                   'Emax_lv0', 'Emax_ra', 'Emax_rv0', 'f_ab_max', 'fab_o', 'fall_time_ven', 'fes_inf', 'fes_min',
                   'fes_o', 'fev_inf', 'fev_o', 'GT_s', 'GT_v', 'Io_met', 'Io_sv', 'K2', 'k_ab', 'kcc_sv', 'KE_la',
                   'KE_lv', 'KE_ra', 'KE_rv', 'kes', 'kmet', 'Kv_mi', 'Kv_po', 'Kv_tr', 'l', 'MO2_bp', 'P0_la', 'P0_lv',
                   'P0_ra', 'P0_rv', 'P_n', 'PaCO2_n', 'r', 'R_pa', 'R_pp', 'R_rs', 'R_sa', 'rise_time_atr',
                   'rise_time_ven', 'Rvc_n', 'T0', 'theta_svn', 'V0_dead', 'V_nominal', 'V_scale', 'Vu_amv0', 'Vu_bv',
                   'Vu_ev0', 'Vu_jp', 'Vu_la', 'Vu_lv', 'Vu_ra', 'Vu_rv', 'Vu_sv0', 'Wb_sh', 'Wb_sv'
}

# Build parameter_idx: indices into the full 230-param space for subset_vars
# (sorted in sp["names"] order, matching what the workflow does)
parameter_idx = [i for i, name in enumerate(sp["names"]) if name in subset_vars_set]
print(f"parameter_idx has {len(parameter_idx)} entries (expect 63)")

# The 7 sensitive parameters for Max_LA_Volume and their DGSM contributions
SENSITIVE_PARAMS = [
    'a2', 'ahead1', 'beta2', 'C2', 'C_jp', 'C_O2_param1', 'C_sv', 'Cvam_O2_n', 'E_rs', 'Emax_la',
     'Emax_lv0', 'Emax_ra', 'Emax_rv0', 'f_ab_max', 'fab_o', 'fall_time_ven', 'fes_inf', 'fes_min',
     'fes_o', 'fev_inf', 'fev_o', 'GT_s', 'GT_v', 'Io_met', 'Io_sv', 'K2', 'k_ab', 'kcc_sv', 'KE_la',
     'KE_lv', 'KE_ra', 'KE_rv', 'kes', 'kmet', 'Kv_mi', 'Kv_po', 'Kv_tr', 'l', 'MO2_bp', 'P0_la', 'P0_lv',
     'P0_ra', 'P0_rv', 'P_n', 'PaCO2_n', 'r', 'R_pa', 'R_pp', 'R_rs', 'R_sa', 'rise_time_atr',
     'rise_time_ven', 'Rvc_n', 'T0', 'theta_svn', 'V0_dead', 'V_nominal', 'V_scale', 'Vu_amv0', 'Vu_bv',
     'Vu_ev0', 'Vu_jp', 'Vu_la', 'Vu_lv', 'Vu_ra', 'Vu_rv', 'Vu_sv0', 'Wb_sh', 'Wb_sv'
]

# For each sensitive param we need TWO indices:
#   - full_idx: column in X_train (full 230-dim space) — for the x-axis values
#   - calib_idx: column in X_calib (63-dim subset) — to confirm alignment
# We use full_idx to grab the raw parameter value for the x-axis.

sensitive_full_idx = {}
for name in SENSITIVE_PARAMS:
    idx = sp["names"].index(name)
    sensitive_full_idx[name] = idx

print("Sensitive param → full index:")
for name, idx in sensitive_full_idx.items():
    print(f"  {name:15s} → {idx}")
print(f"\nparameter_idx length: {len(parameter_idx)} (expect 63)")

# ── 3. LOAD DATA & EMULATOR ────────────────────────────────────────
X_train = np.load(X_TRAIN_PATH)
Y_train = np.load(Y_TRAIN_PATH)
cols_to_drop = [11, 14, 17, 20, 27, 30]


def drop_columns(arr, cols_to_drop):
    cols_to_drop = sorted(set(cols_to_drop))
    return np.delete(arr, cols_to_drop, axis=1)

Y_train = drop_columns(Y_train, cols_to_drop)

mask = (X_train[:, 199] > 30) #& (Y_train [:, 14] < 90)
paramset = X_train[mask]
Results = Y_train[mask]
param_keys = list(sp["names"])
param_samples = [dict(zip(param_keys, row)) for row in paramset]
print(param_samples[-1])


X_calib = X_train[:, parameter_idx]  # [N, 63] — emulator input
Y_actual = Y_train[:, TARGET_COL]  # [N]     — true Max_LA_Volume

print(f"\nX_train:  {X_train.shape}")
print(f"X_calib:  {X_calib.shape}")
print(f"Y_actual: {Y_actual.shape}  range [{Y_actual.min():.2f}, {Y_actual.max():.2f}]")

emulator = joblib.load(EMULATOR_PATH)
print(f"Emulator: {type(emulator).__name__}")

# ── 4. PREDICT ──────────────────────────────────────────────────────
X_calib = torch.as_tensor(X_calib, dtype=torch.float32)

with torch.no_grad():
    mean, var = emulator.predict_mean_and_variance(X_calib)

mean_np = mean.squeeze().cpu().numpy()
std_np = torch.sqrt(var).squeeze().cpu().numpy()
y_np = Y_actual
residuals = mean_np - y_np

# Overall metrics
r2 = r2_score(y_np, mean_np)
rmse = np.sqrt(mean_squared_error(y_np, mean_np))
print(f"\n{'=' * 50}")
print(f"OVERALL  R² = {r2:.4f}  |  RMSE = {rmse:.4f}")
print(f"{'=' * 50}")

# ── 5. PER-PARAMETER PLOTS ─────────────────────────────────────────
n_params = len(SENSITIVE_PARAMS)
fig, axes = plt.subplots(n_params, 2, figsize=(16, 4 * n_params))

for row, (param_name) in enumerate(SENSITIVE_PARAMS):
    full_col = sensitive_full_idx[param_name]
    x_vals = X_train[:, full_col]

    # --- Left: actual & predicted vs parameter value ---
    ax_left = axes[row, 0]
    sort_idx = np.argsort(x_vals)

    ax_left.scatter(x_vals[sort_idx], y_np[sort_idx],
                    s=8, alpha=0.35, color="tab:blue", label="Actual", zorder=2)
    ax_left.scatter(x_vals[sort_idx], mean_np[sort_idx],
                    s=8, alpha=0.35, color="tab:orange", label="Predicted", zorder=3)
    # ±2σ band
    ax_left.fill_between(
        x_vals[sort_idx],
        (mean_np - 2 * std_np)[sort_idx],
        (mean_np + 2 * std_np)[sort_idx],
        alpha=0.12, color="tab:orange", label="±2σ", zorder=1
    )
    ax_left.set_xlabel(param_name, fontsize=11)
    ax_left.set_ylabel("Max_LA_Volume", fontsize=11)
    ax_left.set_title(f"{param_name}", fontsize=12, fontweight="bold")
    ax_left.legend(fontsize=9, loc="best")
    ax_left.grid(True, alpha=0.25)

    # --- Right: residual vs parameter value, coloured by abs(residual) ---
    ax_right = axes[row, 1]
    abs_res = np.abs(residuals)
    sc = ax_right.scatter(x_vals, residuals, c=abs_res, cmap="RdYlGn_r",
                          s=10, alpha=0.5, vmin=0, vmax=np.percentile(abs_res, 95))
    ax_right.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax_right.set_xlabel(param_name, fontsize=11)
    ax_right.set_ylabel("Residual (pred − actual)", fontsize=11)
    ax_right.set_title(f"Residual vs {param_name}", fontsize=12)
    ax_right.grid(True, alpha=0.25)

    # Binned RMSE annotation
    n_bins = 5
    bin_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
    bin_labels = []
    for b in range(n_bins):
        mask = (x_vals >= bin_edges[b]) & (x_vals < bin_edges[b + 1])
        if b == n_bins - 1:  # include right edge in last bin
            mask = (x_vals >= bin_edges[b]) & (x_vals <= bin_edges[b + 1])
        if mask.sum() > 1:
            bin_rmse = np.sqrt(np.mean(residuals[mask] ** 2))
            bin_r2 = r2_score(y_np[mask], mean_np[mask]) if mask.sum() > 2 else float('nan')
            mid = 0.5 * (bin_edges[b] + bin_edges[b + 1])
            ax_right.annotate(
                f"R²={bin_r2:.2f}\nn={mask.sum()}",
                xy=(mid, ax_right.get_ylim()[1]),
                xytext=(mid, ax_right.get_ylim()[1]),
                fontsize=7, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
            )

    plt.colorbar(sc, ax=ax_right, label="|residual|", shrink=0.8)

fig.suptitle(
    f"Max_LA_Volume Emulator Diagnostics  —  Overall R² = {r2:.4f}, RMSE = {rmse:.4f}",
    fontsize=15, fontweight="bold", y=1.005
)
fig.tight_layout()
fig.savefig("Max_LA_Volume_per_param_eval.png", dpi=200, bbox_inches="tight")
# plt.show()
print(f"\nSaved: Max_LA_Volume_per_param_eval.png")

# # ── 6. SUMMARY TABLE ───────────────────────────────────────────────
# print(f"\n{'Param':<15} {'DGSM%':>7} {'Corr(x,resid)':>14} {'Bin RMSE range':>20}")
# print("-" * 60)
# for param_name in SENSITIVE_PARAMS:
#     full_col = sensitive_full_idx[param_name]
#     x_vals = X_train[:, full_col].cpu().numpy()
#     corr = np.corrcoef(x_vals, residuals)[0, 1]
#
#     # binned RMSE spread
#     bin_edges = np.linspace(x_vals.min(), x_vals.max(), 6)
#     bin_rmses = []
#     for b in range(5):
#         mask = (x_vals >= bin_edges[b]) & (x_vals < bin_edges[b + 1])
#         if b == 4:
#             mask = (x_vals >= bin_edges[b]) & (x_vals <= bin_edges[b + 1])
#         if mask.sum() > 1:
#             bin_rmses.append(np.sqrt(np.mean(residuals[mask] ** 2)))
#
#     if bin_rmses:
#         rmse_range = f"[{min(bin_rmses):.2f}, {max(bin_rmses):.2f}]"
#     else:
#         rmse_range = "N/A"
#
#     print(f"{param_name:<15} {corr:>13.3f} {rmse_range:>20}")

# # ── 3. LOAD DATA & EMULATOR ────────────────────────────────────────
# X_train = torch.load(X_TRAIN_PATH, map_location="cpu")
# Y_train = torch.load(Y_TRAIN_PATH, map_location="cpu")
#
# print(f"X_train shape: {X_train.shape}")
# print(f"Y_train shape: {Y_train.shape}")
#
# # Slice to calibration params only
# X_calib = X_train[:, parameter_idx]
# Y_actual = Y_train[:, TARGET_COL]  # Max_LA_Volume column
#
# print(f"X_calib shape (input to emulator): {X_calib.shape}")
# print(f"Y_actual shape: {Y_actual.shape}")
# print(f"Y_actual range: [{Y_actual.min():.2f}, {Y_actual.max():.2f}]")
#
# # Load the saved emulator
# emulator = joblib.load(EMULATOR_PATH)
# print(f"Emulator loaded: {type(emulator)}")
#
# # ── 4. PREDICT ──────────────────────────────────────────────────────
# with torch.no_grad():
#     mean, var = emulator.predict_mean_and_variance(X_calib)
#
# mean = mean.squeeze().cpu().numpy()
# std = torch.sqrt(var).squeeze().cpu().numpy()
# y_actual_np = Y_actual.cpu().numpy()
#
# # ── 5. METRICS ──────────────────────────────────────────────────────
# r2 = r2_score(y_actual_np, mean)
# rmse = np.sqrt(mean_squared_error(y_actual_np, mean))
# print(f"\n=== Max_LA_Volume Emulator Evaluation ===")
# print(f"R²   = {r2:.4f}")
# print(f"RMSE = {rmse:.4f}")
# print(f"Mean predicted uncertainty (std): {std.mean():.4f}")
#
# # ── 6. PLOT ─────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#
# # --- Panel 1: Predicted vs Actual ---
# ax = axes[0]
# lo = min(y_actual_np.min(), mean.min())
# hi = max(y_actual_np.max(), mean.max())
# margin = (hi - lo) * 0.05
#
# ax.errorbar(y_actual_np, mean, yerr=2 * std, fmt='o', alpha=0.3,
#             markersize=3, elinewidth=0.5, capsize=0, label='±2σ')
# ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
#         'r--', linewidth=1.5, label='y = x (perfect)')
# ax.set_xlabel("Actual (simulator)", fontsize=12)
# ax.set_ylabel("Predicted (emulator)", fontsize=12)
# ax.set_title(f"Max_LA_Volume: Predicted vs Actual\nR² = {r2:.4f}  |  RMSE = {rmse:.4f}", fontsize=13)
# ax.legend(fontsize=10)
# ax.set_xlim(lo - margin, hi + margin)
# ax.set_ylim(lo - margin, hi + margin)
# ax.set_aspect('equal', adjustable='box')
# ax.grid(True, alpha=0.3)
#
# # --- Panel 2: Residuals ---
# ax2 = axes[1]
# residuals = mean - y_actual_np
# ax2.scatter(y_actual_np, residuals, alpha=0.4, s=10)
# ax2.axhline(0, color='red', linestyle='--', linewidth=1.5)
# ax2.set_xlabel("Actual (simulator)", fontsize=12)
# ax2.set_ylabel("Residual (predicted − actual)", fontsize=12)
# ax2.set_title("Residuals vs Actual", fontsize=13)
# ax2.grid(True, alpha=0.3)
#
# fig.tight_layout()
# fig.savefig("Max_LA_Volume_emulator_eval.png", dpi=200, bbox_inches="tight")
# plt.show()
# print(f"\nPlot saved to: Max_LA_Volume_emulator_eval.png")
