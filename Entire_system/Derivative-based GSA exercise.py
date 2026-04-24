import math
from collections import OrderedDict

from SALib import ProblemSpec
from SALib.plotting.bar import plot as barplot
from scipy.stats import spearmanr

# from SALib.analyze import dgsm
import dgsm_edited as dgsm
import matplotlib.pyplot as plt
import numpy as np


X = np.load('DGSM_Exercise_Paper/DGSM_500_X_exercise_20_21_04.npy')
Result = np.load('DGSM_Exercise_Paper/DGSM_Result_exercise_20_21_04.npy')

# Result = np.vstack([Result0, Result1, Result3, Result4])
# np.save("DGSM_500_Result_rest_20_10_04.npy", Result)
lower = 0.8
upper = 1.2

Stroke_Volume = Result[:, 3] - Result[:, 4]
Ejection_fraction = (Stroke_Volume / Result[:, 3]) * 100
Result = np.column_stack((Result, Stroke_Volume))
Result = np.column_stack((Result, Ejection_fraction))

D = X.shape[1]
block_size = D + 1
n_blocks = X.shape[0] // block_size
# Find basepoint indices (first row of each block)
base_idx = np.arange(0, X.shape[0], block_size)
# Mask: True if basepoint result != 0
mask_blocks = Result[base_idx, 0] != 0   # check column 0 (e.g. HR); adjust if needed
# OR: drop block if *any* nan appears in that block
mask_blocks_nan = np.array([
    np.all(np.isfinite(Result[i:i+block_size]))   # True if block has no nan
    for i in base_idx
])

# Compute variability (std) within each block
block_std = np.zeros((n_blocks, Result.shape[1]))

for b, i in enumerate(base_idx):
    block = Result[i:i + block_size]
    block_std[b] = np.nanstd(block, axis=0)

for b, i in enumerate(base_idx):
    print(
        f"Block {b:4d} | std = {block_std[b, 3]:.4g}"
        # f"STD: {block_std[b]}"
    )

# Threshold = mean + 3*std of block stds, computed across blocks for each output
std_mean = np.nanmean(block_std, axis=0)
std_std  = np.nanstd(block_std, axis=0)
std_thresh = std_mean + 3 * std_std

# Keep blocks only if ALL output stds are below their respective thresholds
mask_blocks_std = np.all(block_std <= std_thresh, axis=1)

# # Filter base points where E_rs (col 14) and R_rs (col 15) are within +/-20% of nominal values
# E_rs_nominal = 21.9
# R_rs_nominal = 3.02
# E_rs_base = X[base_idx, 14]
# R_rs_base = X[base_idx, 15]
# mask_blocks_E_rs = np.abs(E_rs_base - E_rs_nominal) / E_rs_nominal <= 0.40
# mask_blocks_R_rs = np.abs(R_rs_base - R_rs_nominal) / R_rs_nominal <= 0.40
#
# Keep only blocks where all perturbed HR values are within 0.03 of the base HR (convergence check)
HR_col = 0
mask_blocks_conv = np.array([
    np.all(np.abs(Result[i + 1:i + block_size, HR_col] - Result[i, HR_col]) < 0.03)
    for i in base_idx
])

HR_col = 25
mask_blocks_conv_tidal = np.array([
    np.all(np.abs(Result[i + 1:i + block_size, HR_col] - Result[i, HR_col]) < 0.03)
    for i in base_idx
])

mask_blocks = mask_blocks & mask_blocks_nan & mask_blocks_conv & mask_blocks_std #& mask_blocks_conv_tidal#& mask_blocks_std # & mask_blocks_E_rs & mask_blocks_R_rs # & mask_blocks_std
print(np.count_nonzero(mask_blocks))
# Expand mask to all rows in a block
mask_full = np.repeat(mask_blocks, block_size)


# Filter arrays
X = X[mask_full]
Result = Result[mask_full]

HR = Result[:, 10]

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
            [0.03222905099 * lower, 0.03222905099 * upper],  # beta2 [MAP]
            [72.97384644 * lower, 72.97384644 * upper],  # C2 [MAP]
            [161.8709412 * lower, 161.8709412 * upper],  # K2 [MAP]
            [2.114250183 * lower, 2.114250183 * upper],  # a2 [MAP]
            [0.05591 * lower, 0.05591 * upper],  # alpha2
            [346000 * lower, 346000 * upper],  # KCCO2
            [0.1698 * lower, 0.1698 * upper],  # GV_dead
            [0.2332 * lower, 0.2332 * upper],  # KcCO2
            [1 * lower, 1 * upper],  # KcMRV
            [0.2025 * lower, 0.2025 * upper],  # KpCO2
            [4.72e-09 * lower, 4.72e-09 * upper],  # KpO2
            [0.1370924413 * lower, 0.1370924413 * upper],  # V0_dead [MAP]
            [0.0673 * lower, 0.0673 * upper],  # VA_rest
            [20.46212959 * 0.8, 20.46212959 * 1.2],  # E_rs [MAP]
            [3.383113146 * 0.8, 3.383113146 * 1.2],  # R_rs [MAP]
            [3.125411034 * lower, 3.125411034 * upper],  # C_jp [MAP]
            [0.28 * lower, 0.28 * upper],  # C_sa
            [0.00022 * lower, 0.00022 * upper],  # L_sa
            [0.05399359763 * lower, 0.05399359763 * upper],  # R_sa [MAP]
            [9.4 * lower, 9.4 * upper],  # C_amv
            [10.71 * lower, 10.71 * upper],  # C_bv
            [20 * lower, 20 * upper],  # C_ev
            [3.57 * lower, 3.57 * upper],  # C_hv
            [6.28 * lower, 6.28 * upper],  # C_rmv
            [55.66192627 * lower, 55.66192627 * upper],  # C_sv [MAP]
            [24.17 * lower, 24.17 * upper],  # kr_am
            [10 * lower, 10 * upper],  # P_0
            [0.0833 * lower, 0.0833 * upper],  # R_amv_n
            [0.075 * lower, 0.075 * upper],  # R_bv_n
            [0.04 * lower, 0.04 * upper],  # R_ev_n
            [0.224 * lower, 0.224 * upper],  # R_hv_n
            [0.125 * lower, 0.125 * upper],  # R_rmv_n
            [0.038 * lower, 0.038 * upper],  # R_sv_n
            [0.15 * lower, 0.15 * upper],  # K1_vc
            [0.3855 * lower, 0.3855 * upper],  # D1
            [50 * lower, 50 * upper],  # Vvc_min
            [10000 * lower, 10000 * upper],  # Kr_vc
            [0.02386650629 * lower, 0.02386650629 * upper],  # Rvc_n [MAP]
            [0.76 * lower, 0.76 * upper],  # C_pa
            [5.8 * lower, 5.8 * upper],  # C_pp
            [25.37 * lower, 25.37 * upper],  # C_pv
            [0.00018 * lower, 0.00018 * upper],  # L_pa
            [0.02044834569 * lower, 0.02044834569 * upper],  # R_pa [MAP]
            [0.08207768947 * lower, 0.08207768947 * upper],  # R_pp [MAP]
            [0.0056 * lower, 0.0056 * upper],  # R_pv
            [0.4053699076 * lower, 0.4053699076 * upper],  # Emax_la [MAP]
            [0.4931329489 * lower, 0.4931329489 * upper],  # P0_la [MAP]
            [0.4157464504 * lower, 0.4157464504 * upper],  # Emax_ra [MAP]
            [0.3835878372 * lower, 0.3835878372 * upper],  # P0_ra [MAP]
            [0.05612468719 * lower, 0.05612468719 * upper],  # KE_la [MAP]
            [0.05498290807 * lower, 0.05498290807 * upper],  # KE_ra [MAP]
            [1.599273324 * lower, 1.599273324 * upper],  # P0_lv [MAP]
            [1.373883843 * lower, 1.373883843 * upper],  # P0_rv [MAP]
            [0.04 * lower, 0.04 * upper],  # s
            [21.9467907 * lower, 21.9467907 * upper],  # fab_o [MAP]
            [15.4826498 * lower, 15.4826498 * upper],  # fes_o [MAP]
            [2.157571554 * lower, 2.157571554 * upper],  # fes_inf [MAP]
            [80 * lower, 80 * upper],  # fes_max
            [3.641415596 * lower, 3.641415596 * upper],  # fev_o [MAP]
            [5.535371304 * lower, 5.535371304 * upper],  # fev_inf [MAP]
            [0.06547223032 * lower, 0.06547223032 * upper],  # kes [MAP]
            [7.06 * lower, 7.06 * upper],  # kev
            [0.658 * lower, 0.658 * upper],  # Io_sh
            [0.65 * lower, 0.65 * upper],  # Io_sp
            [0.4148634076 * lower, 0.4148634076 * upper],  # Io_sv [MAP]
            [0.126 * lower, 0.126 * upper],  # Io_v
            [0.114 * lower, 0.114 * upper],  # kcc_sh
            [0.13 * lower, 0.13 * upper],  # kcc_sp
            [0.101082772 * lower, 0.101082772 * upper],  # kcc_sv [MAP]
            [0.0162 * lower, 0.0162 * upper],  # kcc_v
            [9 * lower, 9 * upper],  # Ysh_max
            [-0.0283 * upper, -0.0283 * lower],  # Ysh_min
            [5.5 * lower, 5.5 * upper],  # Ysp_max
            [-0.037 * upper, -0.037 * lower],  # Ysp_min
            [64.9 * lower, 64.9 * upper],  # Ysv_max
            [-0.437 * upper, -0.437 * lower],  # Ysv_min
            [1.9 * lower, 1.9 * upper],  # Yv_max
            [-0.0008 * upper, -0.0008 * lower],  # Yv_min
            [-0.68 * upper, -0.68 * lower],  # theta_v
            [-1.789494157 * upper, -1.789494157 * lower],  # Wb_sh [MAP]
            [-1.1375 * upper, -1.1375 * lower],  # Wb_sp
            [-1.000464439 * upper, -1.000464439 * lower],  # Wb_sv [MAP]
            [1 * lower, 1 * upper],  # Wc_sh
            [1.716 * lower, 1.716 * upper],  # Wc_sp
            [1.716 * lower, 1.716 * upper],  # Wc_sv
            [0.2 * lower, 0.2 * upper],  # Wc_v
            [-0.3997 * upper, -0.3997 * lower],  # Wp_sp
            [-0.3997 * upper, -0.3997 * lower],  # Wp_sv
            [-0.103 * upper, -0.103 * lower],  # Wp_v
            [0.4 * lower, 0.4 * upper],  # Wt_sh
            [0.4 * lower, 0.4 * upper],  # Wt_sp
            [0.4 * lower, 0.4 * upper],  # Wt_sv
            [0.4 * lower, 0.4 * upper],  # Wt_v
            [2.370944023 * lower, 2.370944023 * upper],  # Emax_lv0 [MAP]
            [1.16930747 * lower, 1.16930747 * upper],  # Emax_rv0 [MAP]
            [2.658550978 * lower, 2.658550978 * upper],  # fes_min [MAP]
            [0.475 * lower, 0.475 * upper],  # GEmax_lv
            [0.282 * lower, 0.282 * upper],  # GEmax_rv
            [2.47 * lower, 2.47 * upper],  # GR_amp
            [1.94 * lower, 1.94 * upper],  # GR_ep
            [2.47 * lower, 2.47 * upper],  # GR_rmp
            [0.695 * lower, 0.695 * upper],  # GR_sp
            [-58.29 * upper, -58.29 * lower],  # GV_amv
            [-74.21 * upper, -74.21 * lower],  # GV_ev
            [-58.29 * upper, -58.29 * lower],  # GV_rmv
            [-265.4 * upper, -265.4 * lower],  # GV_sv
            [3.51 * lower, 3.51 * upper],  # R_amp0
            [1.655 * lower, 1.655 * upper],  # R_ep0
            [5.27 * lower, 5.27 * upper],  # R_rmp0
            [2.49 * lower, 2.49 * upper],  # R_sp0
            [1 * lower, 1 * upper],  # g_ccsh
            [1.5 * lower, 1.5 * upper],  # g_ccsp
            [6 * lower, 6 * upper],  # kisc_sh
            [2 * lower, 2 * upper],  # kisc_sp
            [2 * lower, 2 * upper],  # kisc_sv
            [45 * lower, 45 * upper],  # PO2_sh
            [30 * lower, 30 * upper],  # PO2_sp
            [30 * lower, 30 * upper],  # PO2_sv
            [3.6 * lower, 3.6 * upper],  # theta_shn
            [13.32 * lower, 13.32 * upper],  # theta_spn
            [11.36718178 * lower, 11.36718178 * upper],  # theta_svn [MAP]
            [53 * lower, 53 * upper],  # x_sh
            [6 * lower, 6 * upper],  # x_sp
            [6 * lower, 6 * upper],  # x_sv
            [42.11872864 * lower, 42.11872864 * upper],  # PaCO2_n [MAP]
            [42.99484253 * lower, 42.99484253 * upper],  # f_ab_max [MAP]
            [2.52 * lower, 2.52 * upper],  # f_ab_min
            [9.596496582 * lower, 9.596496582 * upper],  # k_ab [MAP]
            [94.99234009 * lower, 94.99234009 * 1.05],  # P_n [MAP]
            [112 * 0.9, 112 * upper],  # P_n_max
            [1.4 * lower, 1.4 * upper],  # f_acCO2_n
            [12.3 * lower, 12.3 * upper],  # f_ac_max
            [0.835 * lower, 0.835 * upper],  # f_ac_min
            [29.27 * lower, 29.27 * upper],  # k_ac
            [3 * lower, 3 * upper],  # K_H
            [45 * lower, 45 * upper],  # PaO2_ac_n
            [11.76 * lower, 11.76 * upper],  # G_ap
            [-0.1113971844 * upper, -0.1113971844 * lower],  # GT_s [MAP]
            [0.0733776018 * lower, 0.0733776018 * upper],  # GT_v [MAP]
            [0.6158705354 * lower, 0.6158705354 * upper],  # T0 [MAP]
            [20.9 * lower, 20.9 * upper],  # A
            [92.8 * lower, 92.8 * upper],  # B
            [10570 * lower, 10570 * upper],  # C
            [-5.251 * upper, -5.251 * lower],  # D
            [0.14 * lower, 0.14 * upper],  # Cvb_O2_n
            [10 * lower, 10 * upper],  # gb_O2
            [0.948317647 * lower, 0.948317647 * upper],  # MO2_bp [MAP]
            [6.57 * lower, 6.57 * upper],  # R_bpn
            [0.11 * lower, 0.11 * upper],  # Cvh_O2_n
            [0.155 * lower, 0.155 * upper],  # Cvrm_O2_n
            [35 * lower, 35 * upper],  # gh_O2
            [30 * lower, 30 * upper],  # grm_O2
            [11.11 * lower, 11.11 * upper],  # Kh_CO2
            [142.8 * lower, 142.8 * upper],  # Krm_CO2
            [0.4 * lower, 0.4 * upper],  # MO2_hpn
            [0.86 * lower, 0.86 * upper],  # MO2_rmp
            [19.71 * lower, 19.71 * upper],  # R_hpn
            [12660 * lower, 12660 * upper],  # W_hn
            [0.1522061974 * lower, 0.1522061974 * upper],  # Cvam_O2_n [MAP]
            [30 * lower, 30 * upper],  # gam_O2
            [40 * lower, 40 * upper],  # gM
            [0.3927663863 * lower, 0.3927663863 * upper],  # Io_met [MAP]
            [0.1622164398 * lower, 0.1622164398 * upper],  # kmet [MAP]
            [0.516 * lower, 0.516 * upper],  # MO2_ampn
            [20 * lower, 20 * upper],  # phi_max
            [-1.87 * upper, -1.87 * lower],  # phi_min
            [1000 * lower, 1000 * upper],  # Kp_ao
            [5000 * lower, 5000 * upper],  # Kf_ao
            [2 * lower, 2 * upper],  # Kb_ao
            [7 * lower, 7 * upper],  # Kv_ao
            [1.309 * lower, 1.309 * upper],  # theta_ao_max
            [1200 * lower, 1200 * upper],  # Kp_mi
            [200 * lower, 200 * upper],  # Kf_mi
            [2 * lower, 2 * upper],  # Kb_mi
            [3.232190371 * lower, 3.232190371 * upper],  # Kv_mi [MAP]
            [1.309 * lower, 1.309 * upper],  # theta_mi_max
            [2000 * lower, 2000 * upper],  # Kp_po
            [2000 * lower, 2000 * upper],  # Kf_po
            [2 * lower, 2 * upper],  # Kb_po
            [5.827977657 * lower, 5.827977657 * upper],  # Kv_po [MAP]
            [1.309 * lower, 1.309 * upper],  # theta_po_max
            [2000 * lower, 2000 * upper],  # Kp_tr
            [200 * lower, 200 * upper],  # Kf_tr
            [2 * lower, 2 * upper],  # Kb_tr
            [3.484293938 * lower, 3.484293938 * upper],  # Kv_tr [MAP]
            [1.309 * lower, 1.309 * upper],  # theta_tr_max
            [0.0000317 * lower, 0.0000317 * upper],  # alpha_O2
            [350 * lower, 350 * upper],  # R_po
            [400 * lower, 400 * upper],  # R_mi
            [400 * lower, 400 * upper],  # R_tr
            [350 * lower, 350 * upper],  # R_ao
            [0.00147458876 * lower, 0.00147458876 * upper],  # C_O2_param1 [MAP]
            [2.6 * lower, 2.6 * upper],  # C_O2_param2
            [3.03e-5 * lower, 3.03e-5 * upper],  # C_O2_param3
            [104 * lower, 104 * upper],  # PAMO2_nominal
            [261.2475281 * lower, 261.2475281 * upper],  # Vu_bv [MAP]
            [93.16 * lower, 93.16 * upper],  # Vu_hv
            [520.1243286 * lower, 520.1243286 * upper],  # Vu_jp [MAP]
            [123 * lower, 123 * upper],  # Vu_vc
            [116.68 * lower, 116.68 * upper],  # Vu_pp
            [114 * lower, 114 * upper],  # Vu_pv
            [25.18439865 * lower, 25.18439865 * upper],  # Vu_la [MAP]
            [18.5427227 * lower, 18.5427227 * upper],  # Vu_lv [MAP]
            [35.5100708 * lower, 35.5100708 * upper],  # Vu_ra [MAP]
            [42.27091217 * lower, 42.27091217 * upper],  # Vu_rv [MAP]
            [8 * lower, 8 * upper],  # tau_Emax_lv
            [8 * lower, 8 * upper],  # tau_Emax_rv
            [2 * lower, 2 * upper],  # tau_Ramp
            [2 * lower, 2 * upper],  # tau_Rep
            [2 * lower, 2 * upper],  # tau_Rrmp
            [2 * lower, 2 * upper],  # tau_Rsp
            [20 * lower, 20 * upper],  # tau_Vamv
            [20 * lower, 20 * upper],  # tau_Vev
            [20 * lower, 20 * upper],  # tau_Vrmv
            [20 * lower, 20 * upper],  # tau_Vsv
            [249.5251617 * lower, 249.5251617 * upper],  # Vu_amv0 [MAP]
            [539.7653809 * lower, 539.7653809 * upper],  # Vu_ev0 [MAP]
            [190.95 * lower, 190.95 * upper],  # Vu_rmv0
            [1277.881592 * lower, 1277.881592 * upper],  # Vu_sv0 [MAP]
            [20 * lower, 20 * upper],  # tau_cc
            [30 * lower, 30 * upper],  # tau_isc
            [2.076 * lower, 2.076 * upper],  # tau_p
            [0.8 * lower, 0.8 * upper],  # tau_z
            [2 * lower, 2 * upper],  # tau_ac
            [2 * lower, 2 * upper],  # tau_ap
            [2 * lower, 2 * upper],  # tau_Ts
            [1.5 * lower, 1.5 * upper],  # tau_Tv
            [20 * lower, 20 * upper],  # tau_CO2
            [10 * lower, 10 * upper],  # tau_O2
            [5 * lower, 5 * upper],  # tau_w
            [40 * lower, 40 * upper],  # tau_M
            [10 * lower, 10 * upper],  # tau_met
            [2 * lower, 2 * upper],  # DEmax_lv
            [2 * lower, 2 * upper],  # DEmax_rv
            [2 * lower, 2 * upper],  # DR_amp
            [2 * lower, 2 * upper],  # DR_ep
            [2 * lower, 2 * upper],  # DR_rmp
            [2 * lower, 2 * upper],  # DR_sp
            [5 * lower, 5 * upper],  # DV_amv
            [5 * lower, 5 * upper],  # DV_ev
            [5 * lower, 5 * upper],  # DV_rmv
            [5 * lower, 5 * upper],  # DV_sv
            [2 * lower, 2 * upper],  # DT_s
            [0.2 * lower, 0.2 * upper],  # DT_v
            [4 * lower, 4 * upper],  # Dmet
            [0.3 * lower, 0.3 * upper],  # Ta
            [0.01238193363 * lower, 0.01238193363 * upper],  # KE_lv [MAP]
            [0.01171110664 * lower, 0.01171110664 * upper],  # KE_rv [MAP]
            [0.1 * lower, 0.1 * upper],  # T1
            [0.2 * lower, 0.2 * upper],  # T2
            [3 * lower, 3 * upper],  # VL_CO2
            [2.5 * lower, 2.5 * upper],  # VL_O2
            [20 * lower, 20 * upper],  # KCSFCO2
            [0.01 * lower, 0.01 * upper],  # VB
            [50 * lower, 50 * upper],  # tauMR
            [0.25 * lower, 0.25 * upper],  # VTCO2
            [0.25 * lower, 0.25 * upper],  # VTO2
            [50 * lower, 50 * upper],  # tau_MRV
            [4.9 * lower, 4.9 * upper],  # scale_param1
            [0.3 * lower, 0.3 * upper],  # scale_param3
            [26.6 * lower, 26.6 * upper],  # scale_param4
            [0.04 * lower, 0.04 * upper],  # scale_param6
            [80 * lower, 80 * upper],  # Pa_O2_lower
            [0.04482196271 * lower, 0.04482196271 * upper],  # rise_time_atr [MAP]
            [0.3259368539 * 0.8, 0.3259368539 * 1.2],  # rise_time_ven [MAP]
            [0.5003861189 * 0.85, 0.5003861189 * 1.15],  # fall_time_ven [MAP]
            [0.8912405968 * 0.92, 0.8912405968 * 1.08],  # ahead1 [MAP]
            [0.0873 * lower, 0.0873 * upper],  # theta_min
            [1.194072247 * 0.85, 1.194072247 * 1.15],  # r [MAP]
            [1.307331085 * 0.85, 1.307331085 * 1.15],  # l [MAP]
            [139.8052521 * lower, 139.8052521 * upper],  # V_nominal [MAP]
            [51.30523682 * lower, 51.30523682 * upper]   # V_scale [MAP]
        ]
    })

output_names = [
    "Heart Rate", "Systolic Pressure", "Diastolic Pressure", "EDV", "ESV",
    "Max RV Volume", "Min RV Volume", "Max RV Pressure", "Min RV Pressure",
    "Min RA Volume", "Max RA Volume", "Min RA Pressure A descent", "Max RA Pressure Atrial contraction",
    "Max RA Pressure Tricuspid Opening", "Min RA Pressure V descent",
    "Min LA Volume", "Max LA Volume", "Min LA Pressure A descent", "Max LA Pressure Atrial contraction",
    "Max LA Pressure Mitral Opening", "Min LA Pressure V descent",
    "LA Contraction Volume diff", "RA Contraction Volume diff", "LV Pressure Deriv", "RV Pressure Deriv", "Tidal Volume", "Minute Ventilation",
    "Cardiac Output", "PaO2", "PaCO2", "Percentage Volume Change",
    "Stroke Volume", "Ejection Fraction"]



# dgsm_summary = OrderedDict()
#
# # min_frac = 0.003  # 0.3%
# min_frac = 0.01
# # min_frac = 0.0
#
# for col in range(Result.shape[1]):
#
#     Y = Result[:, col]
#     output_label = output_names[col]
#
#     # Skip invalid outputs
#     if not np.all(np.isfinite(Y)):
#         print(f"Skipping {output_label} (non-finite values)")
#         continue
#
#     # DGSM analysis
#     Si = dgsm.analyze(sp, X, Y, print_to_console=False)
#
#     dgsm_vals = np.array(Si["dgsm"], dtype=float)
#     names = np.array(Si["names"])
#
#     # Sort descending
#     order = np.argsort(dgsm_vals)[::-1]
#     dgsm_sorted = dgsm_vals[order]
#     names_sorted = names[order]
#
#     total = dgsm_sorted.sum()
#     if total <= 0 or not np.isfinite(total):
#         print(f"Skipping {output_label} (non-positive/invalid DGSM total: {total})")
#         continue
#
#     # --- NEW: keep only params contributing >= 0.3% of total ---
#     thresh = min_frac * total
#     keep = dgsm_sorted >= thresh
#     dgsm_kept = dgsm_sorted[keep]
#     names_kept = names_sorted[keep]
#
#     # Cumulative sensitivity over kept params, but target is 90% of ORIGINAL total
#     cumu = np.cumsum(dgsm_kept)
#
#     target = 0.9 * total
#     if cumu.size == 0:
#         idx_90 = 0
#         top_names_90 = np.array([])
#         top_dgsm_90 = np.array([])
#         reached = 0.0
#     else:
#         idx_90 = np.searchsorted(cumu, target) + 1
#         # if you can't reach 90% without including <0.3% params, cap at all kept
#         idx_90 = min(idx_90, len(cumu))
#         top_names_90 = names_kept[:idx_90]
#         top_dgsm_90 = dgsm_kept[:idx_90]
#         reached = cumu[idx_90 - 1] / total
#
#     dgsm_summary[output_label] = {
#         "n_params_90": idx_90,
#         "param_names": top_names_90,
#         "dgsm_values": top_dgsm_90,
#         "min_frac": min_frac,
#         "fraction_of_total_reached": reached,
#         "n_params_passing_0p3pct": int(keep.sum()),
#     }
#
#     # ---- Console output ----
#     print("\n" + "=" * 80)
#     print(f"Output: {output_label}")
#     print(f"Min per-parameter contribution: {min_frac*100:.1f}% (DGSM >= {thresh:.4e})")
#     print(f"Parameters selected (up to 90% total, with cutoff): {idx_90}")
#     print(f"Fraction of total DGSM reached: {reached*100:.2f}%")
#     if reached < 0.90:
#         print("NOTE: Could not reach 90% without including parameters < 1.0% each.")
#
#     print("-" * 80)
#     for name, val in zip(top_names_90, top_dgsm_90):
#         print(f"{name:25s} : {val:.4e}  ({val/total*100:.3f}%)")


from collections import OrderedDict
import math
import numpy as np
import matplotlib.pyplot as plt

dgsm_by_output = {}
for j, out_name in enumerate(output_names):
    Y = Result[:, j]
    Si = dgsm.analyze(sp, X, Y, print_to_console=False)
    dgsm_by_output[out_name] = {
        "dgsm": np.asarray(Si["dgsm"], dtype=float),
        "conf": np.asarray(Si["dgsm_conf"], dtype=float),
        "names": np.asarray(Si["names"]),
    }

coverage = 0.9
min_frac = 0.01   # at least 1% of original total
n_cols = 7
n_out = len(output_names)
n_rows = math.ceil(n_out / n_cols)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4.8 * n_cols, 3.6 * n_rows),
    constrained_layout=True
)
axes = np.atleast_1d(axes).ravel()

dgsm_summary = OrderedDict()

for ax, out_name in zip(axes, output_names):
    dg = dgsm_by_output[out_name]["dgsm"]
    cf = dgsm_by_output[out_name]["conf"]
    pn = dgsm_by_output[out_name]["names"]

    # sort descending
    order = np.argsort(dg)[::-1]
    dg_sorted = dg[order]
    cf_sorted = cf[order]
    pn_sorted = pn[order]

    # keep only finite, positive DGSM
    good = np.isfinite(dg_sorted) & (dg_sorted > 0) & np.isfinite(cf_sorted)
    dg_sorted = dg_sorted[good]
    cf_sorted = cf_sorted[good]
    pn_sorted = pn_sorted[good]

    total = np.sum(dg_sorted)
    if total <= 0 or not np.isfinite(total):
        ax.text(0.5, 0.5, "No finite DGSM", ha="center", va="center")
        ax.set_title(out_name, fontsize=10)
        ax.axis("off")
        continue

    # enforce minimum 1% of ORIGINAL total
    thresh = min_frac * total
    keep = dg_sorted >= thresh

    dg_kept = dg_sorted[keep]
    cf_kept = cf_sorted[keep]
    pn_kept = pn_sorted[keep]

    if dg_kept.size == 0:
        ax.text(0.5, 0.5, "No params >= 1%", ha="center", va="center")
        ax.set_title(out_name, fontsize=10)
        ax.axis("off")
        dgsm_summary[out_name] = {
            "n_params_90": 0,
            "param_names": np.array([]),
            "dgsm_values": np.array([]),
            "conf_values": np.array([]),
            "fraction_of_total_reached": 0.0,
            "n_params_passing_cutoff": 0,
            "min_frac": min_frac,
        }
        continue

    # cumulative DGSM of kept params, but relative to ORIGINAL total
    cumu = np.cumsum(dg_kept)
    target = coverage * total

    idx_90 = np.searchsorted(cumu, target) + 1
    idx_90 = min(idx_90, len(cumu))   # cap if 90% cannot be reached

    dg_keep = dg_kept[:idx_90]
    cf_keep = cf_kept[:idx_90]
    pn_keep = pn_kept[:idx_90]

    reached = cumu[idx_90 - 1] / total
    note = ""
    if reached < coverage:
        note = f"\nReached {reached*100:.1f}% only"

    # plot (reverse so biggest is at top)
    y = np.arange(len(dg_keep))
    ax.barh(y, dg_keep[::-1], xerr=cf_keep[::-1])
    ax.set_yticks(y)
    ax.set_yticklabels(pn_keep[::-1], fontsize=7)
    ax.set_title(
        f"{out_name}\n{coverage*100:.0f}% DGSM, params >= {min_frac*100:.0f}% ({len(dg_keep)} params){note}",
        fontsize=10
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    dgsm_summary[out_name] = {
        "n_params_90": len(dg_keep),
        "param_names": pn_keep,
        "dgsm_values": dg_keep,
        "conf_values": cf_keep,
        "fraction_of_total_reached": reached,
        "n_params_passing_cutoff": int(keep.sum()),
        "min_frac": min_frac,
    }

# turn off unused subplot axes
for ax in axes[n_out:]:
    ax.axis("off")

# x-label only on bottom row
for r in range(n_rows):
    for c in range(n_cols):
        k_ax = r * n_cols + c
        if k_ax < n_out and r == n_rows - 1:
            axes[k_ax].set_xlabel("DGSM")

# plt.show()
plt.savefig("DGSM_20.png", dpi=300, bbox_inches="tight")
plt.close()





import numpy as np
import matplotlib.pyplot as plt

def topk_dgsm(sp, X, y, k=5):
    Si = dgsm.analyze(sp, X, y, print_to_console=False)
    dg = np.asarray(Si["dgsm"], dtype=float)
    nm = np.asarray(Si["names"])
    cf = np.asarray(Si["dgsm_conf"], dtype=float)

    good = np.isfinite(dg) & (dg > 0) & np.isfinite(cf)
    dg, nm, cf = dg[good], nm[good], cf[good]

    order = np.argsort(dg)[::-1]
    return nm[order][:k].tolist(), dg[order][:k], cf[order][:k]

k = 5
HR = Result[:, 0]
MV = Result[:, 25]  # adjust if needed

nm_hr, dg_hr, cf_hr = topk_dgsm(sp, X, HR, k)
nm_mv, dg_mv, cf_mv = topk_dgsm(sp, X, MV, k)

# x positions: 0..4 for HR, gap, then 6.5..10.5 for MV
gap = 0.5
x_hr = np.arange(k)
x_mv = np.arange(k) + (k + gap)

# # Set global style
# plt.rcParams.update({
#     "font.size": 20,  # Larger font
#     "font.weight": "bold",  # Bold text
#     # "axes.labelweight": "bold",
#     "axes.titlesize": 16,
#     # "axes.titleweight": "bold",
#     "legend.fontsize": 18,
#     "lines.linewidth": 3.5,  # Thicker lines
# })

plt.figure(figsize=(12, 3.2))

plt.bar(x_hr, dg_hr, yerr=cf_hr, width=0.8, color="tab:blue", label="Heart Rate")
plt.bar(x_mv, dg_mv, yerr=cf_mv, width=0.8, color="tab:red",  label="Minute Ventilation")
ax = plt.gca()
ax.margins(x=0.01)

# ticks and labels (5 left + 5 right)
x_all = np.concatenate([x_hr, x_mv])
labels = nm_hr + nm_mv
plt.xticks(x_all, labels, rotation=25, ha="right")

plt.ylabel("DGSM")
# plt.grid(axis="y", linestyle="--", alpha=0.5)

# optional separator line between groups
plt.axvline((k - 0.5) + gap/2, linestyle="--", alpha=0.4)

plt.legend(loc="upper left")
plt.tight_layout()
plt.show()








Si = dgsm.analyze(sp, X, HR, print_to_console=True)

## Extract and sort
dgsm1 = np.array(Si['dgsm'])
names = np.array(Si['names'])
conf = np.array(Si['dgsm_conf'])

dgsm_sorted = np.argsort(dgsm1)[::-1]  # descending order
top_dgsm = dgsm1[dgsm_sorted]
top_names = names[dgsm_sorted]
top_conf = conf[dgsm_sorted]

# Split the sorted arrays in half
mid = len(top_dgsm) // 2
dgsm_1, dgsm_2 = top_dgsm[:mid], top_dgsm[mid:]
names_1, names_2 = top_names[:mid], top_names[mid:]
conf_1, conf_2 = top_conf[:mid], top_conf[mid:]

# Plot first half
plt.figure(figsize=(10, 8))
plt.bar(names_1, dgsm_1, yerr=conf_1)
plt.xlabel("Sensitivity Index (DGSM)")
plt.title("HR DGSM")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Plot second half
plt.figure(figsize=(10, 8))
plt.bar(names_2, dgsm_2, yerr=conf_2)
plt.xlabel("Sensitivity Index (DGSM)")
plt.title("HR DGSM")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Calculate cumulative sum and total sum
cumusum = np.cumsum(top_dgsm)
total = cumusum[-1]

# Find the index where cumulative sum reaches 90% of total
threshold_index = np.searchsorted(cumusum, 0.90 * total) + 1  # +1 to include that index

# Get variables contributing to 90% of sensitivity
vars_90 = top_names[:threshold_index]
sens_90 = top_dgsm[:threshold_index]

print(f"Number of variables contributing 90% sensitivity: {threshold_index}")
# print("Variables:")
# for var, sens in zip(vars_90, sens_90):
#     print(f"{var}: {sens}")

# Optional: Plot these variables only
plt.figure(figsize=(10, 6))
plt.bar(vars_90, sens_90)
plt.xlabel("Parameters")
plt.ylabel("DGSM Sensitivity")
plt.title("Parameters contributing 90% of DGSM Sensitivity")
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# ranking convergence compared to final basepoint
def rank_stability_final(problem, X, Y, max_blocks, step=1, top_k=threshold_index):
    """
    Compute DGSM rankings as number of base points increases,
    comparing each to the final ranking.

    Parameters
    ----------
    problem : dict
        SALib problem definition
    X, Y : arrays
        Inputs and outputs
    max_blocks : int
        Maximum number of base points (blocks)
    step : int
        Increment in blocks
    top_k : int
        Number of top parameters to track for overlap stability
    """
    D = problem["num_vars"]
    block_sizes = range(step, max_blocks + 1, step)
    rankings = []

    # First, compute rankings for all block sizes
    for nb in block_sizes:
        N = (D + 1) * nb
        Si = dgsm.analyze(problem, X[:N, :], Y[:N])
        dgsm_vals = np.array(Si['dgsm'])
        rank = np.argsort(dgsm_vals)[::-1][:top_k]
        rankings.append(rank)

    # Use final ranking as reference
    final_rank = rankings[-1]

    corrs = []
    overlaps = []

    for rank in rankings:
        # Spearman correlation with final ranking
        corr, _ = spearmanr(rank, final_rank)
        corrs.append(corr)

        # Top-k overlap
        overlap = len(set(rank) & set(final_rank)) / top_k
        overlaps.append(overlap)

    return block_sizes, corrs, overlaps, rankings


# Example usage
max_blocks = int(len(HR)/block_size)
block_sizes, corrs, overlaps, rankings = rank_stability_final(sp, X, HR, max_blocks=max_blocks, step=10)

# Plot both metrics
plt.figure(figsize=(8,5))
plt.plot(block_sizes, corrs, marker="o", label="Spearman correlation")
plt.plot(block_sizes, overlaps, marker="s", label="Top-k overlap")
plt.xlabel("Number of base points (blocks)")
plt.ylabel("Stability metric")
plt.title("DGSM Rankings with Increasing Base Points Compared to All Samples")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()




