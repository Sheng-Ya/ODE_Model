import math
from collections import OrderedDict

from SALib import ProblemSpec
from SALib.plotting.bar import plot as barplot
from scipy.stats import spearmanr

# from SALib.analyze import dgsm
import dgsm_edited as dgsm
import matplotlib.pyplot as plt
import numpy as np


# X = np.load('DGSM_Exercise_Paper/DGSM_500_X_exercise_20_24_04.npy')
# Result = np.load('DGSM_Exercise_Paper/DGSM_500_Result_exercise_20_24_04.npy')

# X = np.load('DGSM_500_X_exercise_20_11_05_constants.npy')
# Result = np.load('DGSM_500_Result_exercise_20_11_05_constants.npy')

X = np.load('DGSM_500_X_exercise_20_25_05.npy')
Result = np.load("DGSM_500_Result_exercise_20_25_05.npy")
# Result0 = np.load(r'X:\home\project\Result_task_00_exercise.npy')
# Result1 = np.load(r'X:\home\project\Result_task_01_exercise.npy')
# Result2 = np.load(r'X:\home\project\Result_task_02_exercise.npy')
# Result3 = np.load(r'X:\home\project\Result_task_03_exercise.npy')
# Result = np.vstack([Result0, Result1, Result2, Result3])
# np.save("DGSM_500_Result_exercise_20_25_05.npy", Result)

# Result2 = np.load('DGSM_Rest_Paper/DGSM_500_Result_rest_20_10_04.npy')
# COLS_TO_DROP = [11, 14, 17, 20, 27, 30]
# Result1 = np.delete(Result, COLS_TO_DROP, axis=1)
# Result2 = np.delete(Result2, COLS_TO_DROP, axis=1)
# Result1 = Result1[::204,0]
# Result2 = Result2[::273,21]

# import seaborn as sns
# plt.figure(figsize=(4,3))
# sns.kdeplot(Result1, bw_method="scott", fill=True, linewidth=1.2)
# # sns.kdeplot(Result2, bw_method="scott", fill=True, linewidth=1.2)
#
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.tight_layout()
# plt.show()


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

# Keep only blocks where all perturbed HR values are within 0.03 of the base HR (convergence check)
HR_col = 0
mask_blocks_conv = np.array([
    np.all(np.abs(Result[i + 1:i + block_size, HR_col] - Result[i, HR_col]) < 0.03)
    for i in base_idx
])

# # Keep only complete DGSM blocks where all chamber volumes are physiological. Didn't change much, only difference in 3 parameters
# vu_la_col = 201
# vu_lv_col = 202
# vu_ra_col = 203
# vu_rv_col = 204
# phys_mask_blocks = np.array([
#     np.all(
#         (Result[i:i + block_size, 4] > X[i:i + block_size, vu_lv_col])
#         & (Result[i:i + block_size, 3] > Result[i:i + block_size, 4])
#         & (Result[i:i + block_size, 6] > X[i:i + block_size, vu_rv_col])
#         & (Result[i:i + block_size, 5] > Result[i:i + block_size, 6])
#         & (Result[i:i + block_size, 9] > X[i:i + block_size, vu_ra_col])
#         & (Result[i:i + block_size, 10] > Result[i:i + block_size, 9])
#         & (Result[i:i + block_size, 15] > X[i:i + block_size, vu_la_col])
#         & (Result[i:i + block_size, 16] > Result[i:i + block_size, 15])
#     )
#     for i in base_idx
# ])

# # Expand mask to all rows in a block
# mask_full = np.repeat(phys_mask_blocks, block_size)
# X = X[mask_full]
# Result = Result[mask_full]

# HR_col = 25
# mask_blocks_conv_tidal = np.array([
#     np.all(np.abs(Result[i + 1:i + block_size, HR_col] - Result[i, HR_col]) < 0.03)
#     for i in base_idx
# ])

mask_blocks = mask_blocks & mask_blocks_nan & mask_blocks_conv & mask_blocks_std # & phys_mask_blocks #& mask_blocks_conv_tidal#& mask_blocks_std # & mask_blocks_E_rs & mask_blocks_R_rs # & mask_blocks_std
print(np.count_nonzero(mask_blocks))
# Expand mask to all rows in a block
mask_full = np.repeat(mask_blocks, block_size)


# Filter arrays
X = X[mask_full]
Result = Result[mask_full]

HR = Result[:, 10]

# sp = ProblemSpec({
#         'names': [
#             # gas
#             "beta2", "C2", "K2", "a2",
#             "alpha2", "KCCO2", "GV_dead",
#             # resp control
#             "KcCO2", "KcMRV", "KpCO2", "KpO2",
#             "V0_dead", "VA_rest",
#             "E_rs", "R_rs",
#             # cardio
#             "C_jp", "C_sa", "L_sa", "R_sa",
#             "C_amv", "C_bv", "C_ev", "C_hv",
#             "C_rmv", "C_sv", "kr_am", "P_0",
#             "R_amv_n", "R_bv_n", "R_ev_n", "R_hv_n",
#             "R_rmv_n", "R_sv_n", "K1_vc", "D1",
#             "Vvc_min", "Kr_vc",
#             "Rvc_n", "C_pa", "C_pp",
#             "C_pv", "L_pa", "R_pa", "R_pp",
#             "R_pv", "Emax_la", "P0_la", "Emax_ra",
#             "P0_ra", "KE_la", "KE_ra", "P0_lv",
#             "P0_rv",
#             "s",
#             # cardio control
#             "fab_o", "fes_o", "fes_inf", "fes_max",
#             "fev_o", "fev_inf", "kes", "kev",
#             "Io_sh", "Io_sp", "Io_sv", "Io_v",
#             "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v",
#             "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min",
#             "Ysv_max", "Ysv_min", "Yv_max", "Yv_min",
#             "theta_v", "Wb_sh", "Wb_sp", "Wb_sv",
#             "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v",
#             "Wp_sp", "Wp_sv", "Wp_v",
#             "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
#             "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv",
#             "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp",
#             "GR_sp", "GV_amv", "GV_ev", "GV_rmv",
#             "GV_sv", "R_amp0", "R_ep0", "R_rmp0",
#             #
#             "R_sp0", "g_ccsh", "g_ccsp",
#             "kisc_sh", "kisc_sp", "kisc_sv",
#             "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn",
#             "theta_spn", "theta_svn", "x_sh", "x_sp",
#             "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min",
#             "k_ab", "P_n", "P_n_max", "f_acCO2_n",
#             "f_ac_max", "f_ac_min", "k_ac", "K_H",
#             "PaO2_ac_n", "G_ap", "GT_s", "GT_v",
#             "T0", "A", "B", "C",
#             "D", "Cvb_O2_n", "gb_O2", "MO2_bp",
#             "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2",
#             "grm_O2", "Kh_CO2", "Krm_CO2", "MO2_hpn",
#             "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n",
#             "gam_O2", "gM", "Io_met", "kmet",
#             "MO2_ampn", "phi_max", "phi_min",
#             # added params
#             "Kp_ao", "Kf_ao", "Kb_ao", "Kv_ao", "theta_ao_max",
#             "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max",
#             "Kp_po", "Kf_po", "Kb_po", "Kv_po", "theta_po_max",
#             "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max",
#             "alpha_O2", "R_po", "R_mi", "R_tr",
#             "R_ao", "C_O2_param1", "C_O2_param2", "C_O2_param3",
#             "PAMO2_nominal", "Vu_bv", "Vu_hv",
#             "Vu_jp", "Vu_vc",
#             "Vu_pp", "Vu_pv", "Vu_la", "Vu_lv",
#             "Vu_ra", "Vu_rv",
#
#             "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp",
#             "tau_Rep", "tau_Rrmp", "tau_Rsp", "tau_Vamv",
#             "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0",
#             "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc",
#             "tau_isc", "tau_p", "tau_z", "tau_ac",
#             "tau_ap", "tau_Ts", "tau_Tv", "tau_CO2",
#             "tau_O2", "tau_w", "tau_M", "tau_met",
#             "DEmax_lv", "DEmax_rv", "DR_amp", "DR_ep",
#             "DR_rmp", "DR_sp", "DV_amv", "DV_ev",
#             "DV_rmv", "DV_sv", "DT_s", "DT_v",
#             "Dmet", "Ta", "KE_lv", "KE_rv",
#             "T1", "T2", "VL_CO2", "VL_O2",
#             "KCSFCO2", "VB", "tauMR", "VTCO2",
#             "VTO2", "tau_MRV",
#
#             # further added
#             "scale_param1", "scale_param3", "scale_param4",
#             "scale_param6", "Pa_O2_lower",
#             "rise_time_atr", "rise_time_ven", "fall_time_ven", "ahead1",
#             "theta_min", "r", "l", "V_nominal", "V_scale"
#         ],
#
#         'bounds': [
#             [0.037361187576 * lower, 0.037361187576 * upper],  # beta2 [MAP]
#             [100.826812355449 * lower, 100.826812355449 * upper],  # C2 [MAP]
#             [169.622481377162 * lower, 169.622481377162 * upper],  # K2 [MAP]
#             [2.036038971916 * lower, 2.036038971916 * upper],  # a2 [MAP]
#             [0.05591 * lower, 0.05591 * upper],
#             [346000 * lower, 346000 * upper],
#             [0.1698 * lower, 0.1698 * upper],
#             [0.2332 * lower, 0.2332 * upper],
#             [1 * lower, 1 * upper],
#             [0.2025 * lower, 0.2025 * upper],
#             [0.00000000472 * lower, 0.00000000472 * upper],
#             [0.18282823609 * lower, 0.18282823609 * upper],  # V0_dead [MAP]
#             [0.0673 * lower, 0.0673 * upper],
#             [18.890244012046 * lower, 18.890244012046 * upper],  # E_rs [MAP]
#             [3.623644350909 * lower, 3.623644350909 * upper],  # R_rs [MAP]
#             [4.06728622462 * lower, 4.06728622462 * upper],  # C_jp [MAP]
#             [0.28 * lower, 0.28 * upper],
#             [0.00022 * lower, 0.00022 * upper],
#             [0.051871492229 * lower, 0.051871492229 * upper],  # R_sa [MAP]
#             [9.4 * lower, 9.4 * upper],
#             [10.71 * lower, 10.71 * upper],
#             [20 * lower, 20 * upper],
#             [3.57 * lower, 3.57 * upper],
#             [6.28 * lower, 6.28 * upper],
#             [52.987192868291 * lower, 52.987192868291 * upper],  # C_sv [MAP]
#             [24.17 * lower, 24.17 * upper],
#             [10 * lower, 10 * upper],
#             [0.0833 * lower, 0.0833 * upper],
#             [0.075 * lower, 0.075 * upper],
#             [0.04 * lower, 0.04 * upper],
#             [0.224 * lower, 0.224 * upper],
#             [0.125 * lower, 0.125 * upper],
#             [0.038 * lower, 0.038 * upper],
#             [0.15 * lower, 0.15 * upper],
#             [0.3855 * lower, 0.3855 * upper],
#             [50 * lower, 50 * upper],
#             [10000 * lower, 10000 * upper],
#             [0.021915411144 * lower, 0.021915411144 * upper],  # Rvc_n [MAP]
#             [0.76 * lower, 0.76 * upper],
#             [5.8 * lower, 5.8 * upper],
#             [25.37 * lower, 25.37 * upper],
#             [0.00018 * lower, 0.00018 * upper],
#             [0.019662904331 * lower, 0.019662904331 * upper],  # R_pa [MAP]
#             [0.077044385521 * lower, 0.077044385521 * upper],  # R_pp [MAP]
#             [0.0056 * lower, 0.0056 * upper],
#             [0.39505525601 * lower, 0.39505525601 * upper],  # Emax_la [MAP]
#             [0.456638725603 * lower, 0.456638725603 * upper],  # P0_la [MAP]
#             [0.385909354649 * lower, 0.385909354649 * upper],  # Emax_ra [MAP]
#             [0.384882977497 * lower, 0.384882977497 * upper],  # P0_ra [MAP]
#             [0.0572715743 * lower, 0.0572715743 * upper],  # KE_la [MAP]
#             [0.042427107208 * lower, 0.042427107208 * upper],  # KE_ra [MAP]
#             [1.715210068467 * lower, 1.715210068467 * upper],  # P0_lv [MAP]
#             [1.274033049278 * lower, 1.274033049278 * upper],  # P0_rv [MAP]
#             [0.04 * lower, 0.04 * upper],
#             [28.185081706614 * lower, 28.185081706614 * upper],  # fab_o [MAP]
#             [14.230002676934 * lower, 14.230002676934 * upper],  # fes_o [MAP]
#             [2.399607677626 * lower, 2.399607677626 * upper],  # fes_inf [MAP]
#             [80 * lower, 80 * upper],
#             [2.770487235522 * lower, 2.770487235522 * upper],  # fev_o [MAP]
#             [7.106351267331 * lower, 7.106351267331 * upper],  # fev_inf [MAP]
#             [0.080996308131 * lower, 0.080996308131 * upper],  # kes [MAP]
#             [7.06 * lower, 7.06 * upper],
#             [0.658 * lower, 0.658 * upper],
#             [0.65 * lower, 0.65 * upper],
#             [0.389235644269 * lower, 0.389235644269 * upper],  # Io_sv [MAP]
#             [0.126 * lower, 0.126 * upper],
#             [0.114 * lower, 0.114 * upper],
#             [0.13 * lower, 0.13 * upper],
#             [0.103576404777 * lower, 0.103576404777 * upper],  # kcc_sv [MAP]
#             [0.0162 * lower, 0.0162 * upper],
#             [9 * lower, 9 * upper],
#             [-0.0283 * upper, -0.0283 * lower],
#             [5.5 * lower, 5.5 * upper],
#             [-0.037 * upper, -0.037 * lower],
#             [64.9 * lower, 64.9 * upper],
#             [-0.437 * upper, -0.437 * lower],
#             [1.9 * lower, 1.9 * upper],
#             [-0.0008 * upper, -0.0008 * lower],
#             [-0.68 * upper, -0.68 * lower],
#             [-1.983796073714 * upper, -1.983796073714 * lower],  # Wb_sh [MAP]
#             [-1.1375 * upper, -1.1375 * lower],
#             [-0.963025739055 * upper, -0.963025739055 * lower],  # Wb_sv [MAP]
#             [1 * lower, 1 * upper],
#             [1.716 * lower, 1.716 * upper],
#             [1.716 * lower, 1.716 * upper],
#             [0.2 * lower, 0.2 * upper],
#             [-0.3997 * upper, -0.3997 * lower],
#             [-0.3997 * upper, -0.3997 * lower],
#             [-0.103 * upper, -0.103 * lower],
#             [0.4 * lower, 0.4 * upper],
#             [0.4 * lower, 0.4 * upper],
#             [0.4 * lower, 0.4 * upper],
#             [0.4 * lower, 0.4 * upper],
#             [2.06151705504 * lower, 2.06151705504 * upper],  # Emax_lv0 [MAP]
#             [1.203022176229 * lower, 1.203022176229 * upper],  # Emax_rv0 [MAP]
#             [2.312410099949 * lower, 2.312410099949 * upper],  # fes_min [MAP]
#             [0.475 * lower, 0.475 * upper],
#             [0.282 * lower, 0.282 * upper],
#             [2.47 * lower, 2.47 * upper],
#             [1.94 * lower, 1.94 * upper],
#             [2.47 * lower, 2.47 * upper],
#             [0.695 * lower, 0.695 * upper],
#             [-58.29 * upper, -58.29 * lower],
#             [-74.21 * upper, -74.21 * lower],
#             [-58.29 * upper, -58.29 * lower],
#             [-265.4 * upper, -265.4 * lower],
#             [3.51 * lower, 3.51 * upper],
#             [1.655 * lower, 1.655 * upper],
#             [5.27 * lower, 5.27 * upper],
#             [2.49 * lower, 2.49 * upper],
#             [1 * lower, 1 * upper],
#             [1.5 * lower, 1.5 * upper],
#             [6 * lower, 6 * upper],
#             [2 * lower, 2 * upper],
#             [2 * lower, 2 * upper],
#             [45 * lower, 45 * upper],
#             [30 * lower, 30 * upper],
#             [30 * lower, 30 * upper],
#             [3.6 * lower, 3.6 * upper],
#             [13.32 * lower, 13.32 * upper],
#             [12.309577661518 * lower, 12.309577661518 * upper],  # theta_svn [MAP]
#             [53 * lower, 53 * upper],
#             [6 * lower, 6 * upper],
#             [6 * lower, 6 * upper],
#             [36.145946392941 * lower, 36.145946392941 * upper],  # PaCO2_n [MAP]
#             [40.911147592035 * lower, 40.911147592035 * upper],  # f_ab_max [MAP]
#             [2.52 * lower, 2.52 * upper],
#             [10.345332082469 * lower, 10.345332082469 * upper],  # k_ab [MAP]
#             [96.593807343782 * lower, 96.593807343782 * 1.05],  # P_n [MAP]
#             [112 * 0.9, 112 * upper],
#             [1.4 * lower, 1.4 * upper],
#             [12.3 * lower, 12.3 * upper],
#             [0.835 * lower, 0.835 * upper],
#             [29.27 * lower, 29.27 * upper],
#             [3 * lower, 3 * upper],
#             [45 * lower, 45 * upper],
#             [11.76 * lower, 11.76 * upper],
#             [-0.115980682677 * upper, -0.115980682677 * lower],  # GT_s [MAP]
#             [0.093398458554 * lower, 0.093398458554 * upper],  # GT_v [MAP]
#             [0.662755467642 * lower, 0.662755467642 * upper],  # T0 [MAP]
#             [20.9 * lower, 20.9 * upper],
#             [92.8 * lower, 92.8 * upper],
#             [10570 * lower, 10570 * upper],
#             [-5.251 * upper, -5.251 * lower],
#             [0.14 * lower, 0.14 * upper],
#             [10 * lower, 10 * upper],
#             [0.806392872949 * lower, 0.806392872949 * upper],  # MO2_bp [MAP]
#             [6.57 * lower, 6.57 * upper],
#             [0.11 * lower, 0.11 * upper],
#             [0.155 * lower, 0.155 * upper],
#             [35 * lower, 35 * upper],
#             [30 * lower, 30 * upper],
#             [11.11 * lower, 11.11 * upper],
#             [142.8 * lower, 142.8 * upper],
#             [0.4 * lower, 0.4 * upper],
#             [0.86 * lower, 0.86 * upper],
#             [19.71 * lower, 19.71 * upper],
#             [12660 * lower, 12660 * upper],
#             [0.136618325734 * lower, 0.136618325734 * upper],  # Cvam_O2_n [MAP]
#             [30 * lower, 30 * upper],
#             [40 * lower, 40 * upper],
#             [0.365243647731 * lower, 0.365243647731 * upper],  # Io_met [MAP]
#             [0.156284159241 * lower, 0.156284159241 * upper],  # kmet [MAP]
#             [0.516 * lower, 0.516 * upper],
#             [20 * lower, 20 * upper],
#             [-1.87 * upper, -1.87 * lower],
#             [1000 * lower, 1000 * upper],
#             [5000 * lower, 5000 * upper],
#             [2 * lower, 2 * upper],
#             [7 * lower, 7 * upper],
#             [1.309 * lower, 1.309 * upper],
#             [1200 * lower, 1200 * upper],
#             [200 * lower, 200 * upper],
#             [2 * lower, 2 * upper],
#             [4.040850924835 * lower, 4.040850924835 * upper],  # Kv_mi [MAP]
#             [1.309 * lower, 1.309 * upper],
#             [2000 * lower, 2000 * upper],
#             [2000 * lower, 2000 * upper],
#             [2 * lower, 2 * upper],
#             [6.032042802243 * lower, 6.032042802243 * upper],  # Kv_po [MAP]
#             [1.309 * lower, 1.309 * upper],
#             [2000 * lower, 2000 * upper],
#             [200 * lower, 200 * upper],
#             [2 * lower, 2 * upper],
#             [3.078695175846 * lower, 3.078695175846 * upper],  # Kv_tr [MAP]
#             [1.309 * lower, 1.309 * upper],
#             [0.0000317 * lower, 0.0000317 * upper],
#             [350 * lower, 350 * upper],
#             [400 * lower, 400 * upper],
#             [400 * lower, 400 * upper],
#             [350 * lower, 350 * upper],
#             [0.001465418486 * lower, 0.001465418486 * upper],  # C_O2_param1 [MAP]
#             [2.6 * lower, 2.6 * upper],
#             [0.0000303 * lower, 0.0000303 * upper],
#             [104 * lower, 104 * upper],
#             [319.120325857796 * lower, 319.120325857796 * upper],  # Vu_bv [MAP]
#             [93.16 * lower, 93.16 * upper],
#             [509.491591784982 * lower, 509.491591784982 * upper],  # Vu_jp [MAP]
#             [123 * lower, 123 * upper],
#             [116.68 * lower, 116.68 * upper],
#             [114 * lower, 114 * upper],
#             [27.289384390602 * lower, 27.289384390602 * upper],  # Vu_la [MAP]
#             [13.641276764031 * lower, 13.641276764031 * upper],  # Vu_lv [MAP]
#             [34.926071035468 * lower, 34.926071035468 * upper],  # Vu_ra [MAP]
#             [43.566591638824 * lower, 43.566591638824 * upper],  # Vu_rv [MAP]
#             [8 * lower, 8 * upper],
#             [8 * lower, 8 * upper],
#             [2 * lower, 2 * upper],
#             [2 * lower, 2 * upper],
#             [2 * lower, 2 * upper],
#             [2 * lower, 2 * upper],
#             [20 * lower, 20 * upper],
#             [20 * lower, 20 * upper],
#             [20 * lower, 20 * upper],
#             [20 * lower, 20 * upper],
#             [253.88464393659 * lower, 253.88464393659 * upper],  # Vu_amv0 [MAP]
#             [522.770609173199 * lower, 522.770609173199 * upper],  # Vu_ev0 [MAP]
#             [190.95 * lower, 190.95 * upper],
#             [1174.878701407525 * lower, 1174.878701407525 * upper],  # Vu_sv0 [MAP]
#             [20 * lower, 20 * upper],
#             [30 * lower, 30 * upper],
#             [2.076 * lower, 2.076 * upper],
#             [0.8 * lower, 0.8 * upper],
#             [2 * lower, 2 * upper],
#             [2 * lower, 2 * upper],
#             [2 * lower, 2 * upper],
#             [1.5 * lower, 1.5 * upper],
#             [20 * lower, 20 * upper],
#             [10 * lower, 10 * upper],
#             [5 * lower, 5 * upper],
#             [40 * lower, 40 * upper],
#             [10 * lower, 10 * upper],
#             [2 * lower, 2 * upper],
#             [2 * lower, 2 * upper],
#             [2 * lower, 2 * upper],
#             [2 * lower, 2 * upper],
#             [2 * lower, 2 * upper],
#             [2 * lower, 2 * upper],
#             [5 * lower, 5 * upper],
#             [5 * lower, 5 * upper],
#             [5 * lower, 5 * upper],
#             [5 * lower, 5 * upper],
#             [2 * lower, 2 * upper],
#             [0.2 * lower, 0.2 * upper],
#             [4 * lower, 4 * upper],
#             [0.3 * lower, 0.3 * upper],
#             [0.012328616147 * lower, 0.012328616147 * upper],  # KE_lv [MAP]
#             [0.012279442473 * lower, 0.012279442473 * upper],  # KE_rv [MAP]
#             [0.1 * lower, 0.1 * upper],
#             [0.2 * lower, 0.2 * upper],
#             [3 * lower, 3 * upper],
#             [2.5 * lower, 2.5 * upper],
#             [20 * lower, 20 * upper],
#             [0.01 * lower, 0.01 * upper],
#             [50 * lower, 50 * upper],
#             [0.25 * lower, 0.25 * upper],
#             [0.25 * lower, 0.25 * upper],
#             [50 * lower, 50 * upper],
#             [4.9 * lower, 4.9 * upper],
#             [0.3 * lower, 0.3 * upper],
#             [26.6 * lower, 26.6 * upper],
#             [0.04 * lower, 0.04 * upper],
#             [80 * lower, 80 * upper],
#             [0.039711933621 * lower, 0.039711933621 * upper],  # rise_time_atr [MAP]
#             [0.343165686803 * lower, 0.343165686803 * upper],  # rise_time_ven [MAP]
#             [0.498695070384 * 0.85, 0.498695070384 * 1.15],  # fall_time_ven [MAP]
#             [0.972301172882 * 0.92, 0.972301172882 * 1.08],  # ahead1 [MAP]
#             [0.0873 * lower, 0.0873 * upper],
#             [1.126938173047 * 0.85, 1.126938173047 * 1.15],  # r [MAP]
#             [1.315543200288 * 0.85, 1.315543200288 * 1.15],  # l [MAP]
#             [134.920729578221 * lower, 134.920729578221 * upper],  # V_nominal [MAP]
#             [44.395890818351 * lower, 44.395890818351 * upper],  # V_scale [MAP]
#         ]
#     })

sp = ProblemSpec({
        'names': [
            'beta2',
            'C2',
            'K2',
            'a2',
            'alpha2',
            'KCCO2',
            'GV_dead',
            'KcCO2',
            'KcMRV',
            'KpCO2',
            'KpO2',
            'V0_dead',
            'VA_rest',
            'E_rs',
            'R_rs',
            'C_jp',
            'C_sa',
            'L_sa',
            'R_sa',
            'C_amv',
            'C_bv',
            'C_ev',
            'C_hv',
            'C_rmv',
            'C_sv',
            'kr_am',
            'P_0',
            'R_amv_n',
            'R_bv_n',
            'R_ev_n',
            'R_hv_n',
            'R_rmv_n',
            'R_sv_n',
            'K1_vc',
            'D1',
            'Vvc_min',
            'Kr_vc',
            'Rvc_n',
            'C_pa',
            'C_pp',
            'C_pv',
            'L_pa',
            'R_pa',
            'R_pp',
            'R_pv',
            'Emax_la',
            'P0_la',
            'Emax_ra',
            'P0_ra',
            'KE_la',
            'KE_ra',
            'P0_lv',
            'P0_rv',
            's',
            'fab_o',
            'fes_o',
            'fes_inf',
            'fes_max',
            'fev_o',
            'fev_inf',
            'kes',
            'kev',
            'Io_sh',
            'Io_sp',
            'Io_sv',
            'Io_v',
            'kcc_sh',
            'kcc_sp',
            'kcc_sv',
            'kcc_v',
            'Ysh_max',
            'Ysh_min',
            'Ysp_max',
            'Ysp_min',
            'Ysv_max',
            'Ysv_min',
            'Yv_max',
            'Yv_min',
            'theta_v',
            'Wb_sh',
            'Wb_sp',
            'Wb_sv',
            'Wc_sh',
            'Wc_sp',
            'Wc_sv',
            'Wc_v',
            'Wp_sp',
            'Wp_sv',
            'Wp_v',
            'Wt_sh',
            'Wt_sp',
            'Wt_sv',
            'Wt_v',
            'Emax_lv0',
            'Emax_rv0',
            'fes_min',
            'GEmax_lv',
            'GEmax_rv',
            'GR_amp',
            'GR_ep',
            'GR_rmp',
            'GR_sp',
            'GV_amv',
            'GV_ev',
            'GV_rmv',
            'GV_sv',
            'R_amp0',
            'R_ep0',
            'R_rmp0',
            'R_sp0',
            'g_ccsh',
            'g_ccsp',
            'kisc_sh',
            'kisc_sp',
            'kisc_sv',
            'PO2_sh',
            'PO2_sp',
            'PO2_sv',
            'theta_shn',
            'theta_spn',
            'theta_svn',
            'x_sh',
            'x_sp',
            'x_sv',
            'PaCO2_n',
            'f_ab_max',
            'f_ab_min',
            'k_ab',
            'P_n',
            'P_n_max',
            'f_acCO2_n',
            'f_ac_max',
            'f_ac_min',
            'k_ac',
            'K_H',
            'PaO2_ac_n',
            'G_ap',
            'GT_s',
            'GT_v',
            'T0',
            'A',
            'B',
            'C',
            'D',
            'Cvb_O2_n',
            'gb_O2',
            'MO2_bp',
            'R_bpn',
            'Cvh_O2_n',
            'Cvrm_O2_n',
            'gh_O2',
            'grm_O2',
            'Kh_CO2',
            'Krm_CO2',
            'MO2_hpn',
            'MO2_rmp',
            'R_hpn',
            'W_hn',
            'Cvam_O2_n',
            'gam_O2',
            'gM',
            'Io_met',
            'kmet',
            'MO2_ampn',
            'phi_max',
            'phi_min',
            'Kp_ao',
            'Kf_ao',
            'Kb_ao',
            'Kv_ao',
            'theta_ao_max',
            'Kp_mi',
            'Kf_mi',
            'Kb_mi',
            'Kv_mi',
            'theta_mi_max',
            'Kp_po',
            'Kf_po',
            'Kb_po',
            'Kv_po',
            'theta_po_max',
            'Kp_tr',
            'Kf_tr',
            'Kb_tr',
            'Kv_tr',
            'theta_tr_max',
            'alpha_O2',
            'R_po',
            'R_mi',
            'R_tr',
            'R_ao',
            'C_O2_param1',
            'C_O2_param2',
            'C_O2_param3',
            'PAMO2_nominal',
            'Vu_bv',
            'Vu_hv',
            'Vu_jp',
            'Vu_vc',
            'Vu_pp',
            'Vu_pv',
            'Vu_la',
            'Vu_lv',
            'Vu_ra',
            'Vu_rv',
            'tau_Emax_lv',
            'tau_Emax_rv',
            'tau_Ramp',
            'tau_Rep',
            'tau_Rrmp',
            'tau_Rsp',
            'tau_Vamv',
            'tau_Vev',
            'tau_Vrmv',
            'tau_Vsv',
            'Vu_amv0',
            'Vu_ev0',
            'Vu_rmv0',
            'Vu_sv0',
            'tau_cc',
            'tau_isc',
            'tau_p',
            'tau_z',
            'tau_ac',
            'tau_ap',
            'tau_Ts',
            'tau_Tv',
            'tau_CO2',
            'tau_O2',
            'tau_w',
            'tau_M',
            'tau_met',
            'DEmax_lv',
            'DEmax_rv',
            'DR_amp',
            'DR_ep',
            'DR_rmp',
            'DR_sp',
            'DV_amv',
            'DV_ev',
            'DV_rmv',
            'DV_sv',
            'DT_s',
            'DT_v',
            'Dmet',
            'Ta',
            'KE_lv',
            'KE_rv',
            'T1',
            'T2',
            'VL_CO2',
            'VL_O2',
            'KCSFCO2',
            'VB',
            'tauMR',
            'VTCO2',
            'VTO2',
            'tau_MRV',
            'scale_param1',
            'scale_param3',
            'scale_param4',
            'scale_param6',
            'Pa_O2_lower',
            'rise_time_atr',
            'rise_time_ven',
            'fall_time_ven',
            'ahead1',
            'theta_min',
            'r',
            'l',
            'V_nominal',
            'V_scale',
        ],

    'bounds': [
        [0.03255 * lower, 0.03255 * upper],
        [98.640209672481 * lower, 98.640209672481 * upper],  # C2 [MAP]
        [168.668058520599 * lower, 168.668058520599 * upper],  # K2 [MAP]
        [2.05720263191 * lower, 2.05720263191 * upper],  # a2 [MAP]
        [0.05591 * lower, 0.05591 * upper],
        [346000 * lower, 346000 * upper],
        [0.1698 * lower, 0.1698 * upper],
        [0.2332 * lower, 0.2332 * upper],
        [1 * lower, 1 * upper],
        [0.2025 * lower, 0.2025 * upper],
        [0.00000000472 * lower, 0.00000000472 * upper],
        [0.180760531763 * lower, 0.180760531763 * upper],  # V0_dead [MAP]
        [0.0673 * lower, 0.0673 * upper],
        [24.771332348072 * lower, 24.771332348072 * upper],  # E_rs [MAP]
        [3.308951736968 * lower, 3.308951736968 * upper],  # R_rs [MAP]
        [3.178371624738 * lower, 3.178371624738 * upper],  # C_jp [MAP]
        [0.28 * lower, 0.28 * upper],
        [0.00022 * lower, 0.00022 * upper],
        [0.067627524644 * lower, 0.067627524644 * upper],  # R_sa [MAP]
        [9.4 * lower, 9.4 * upper],
        [10.71 * lower, 10.71 * upper],
        [20 * lower, 20 * upper],
        [3.57 * lower, 3.57 * upper],
        [6.28 * lower, 6.28 * upper],
        [57.296181668841 * lower, 57.296181668841 * upper],  # C_sv [MAP]
        [24.17 * lower, 24.17 * upper],
        [10 * lower, 10 * upper],
        [0.0833 * lower, 0.0833 * upper],
        [0.075 * lower, 0.075 * upper],
        [0.04 * lower, 0.04 * upper],
        [0.224 * lower, 0.224 * upper],
        [0.125 * lower, 0.125 * upper],
        [0.038 * lower, 0.038 * upper],
        [0.15 * lower, 0.15 * upper],
        [0.3855 * lower, 0.3855 * upper],
        [50 * lower, 50 * upper],
        [10000 * lower, 10000 * upper],
        [0.020011011618 * lower, 0.020011011618 * upper],  # Rvc_n [MAP]
        [5.85 * lower, 5.85 * upper],
        [5.8 * lower, 5.8 * upper],
        [25.37 * lower, 25.37 * upper],
        [0.00018 * lower, 0.00018 * upper],
        [0.023 * lower, 0.023 * upper],
        [0.082324220616 * lower, 0.082324220616 * upper],  # R_pp [MAP]
        [0.0056 * lower, 0.0056 * upper],
        [0.387283349395 * lower, 0.387283349395 * upper],  # Emax_la [MAP]
        [0.51807469575 * lower, 0.51807469575 * upper],  # P0_la [MAP]
        [0.360330241084 * lower, 0.360330241084 * upper],  # Emax_ra [MAP]
        [0.380765929195 * lower, 0.380765929195 * upper],  # P0_ra [MAP]
        [0.057937819808 * lower, 0.057937819808 * upper],  # KE_la [MAP]
        [0.04293465892 * lower, 0.04293465892 * upper],  # KE_ra [MAP]
        [1.442691243336 * lower, 1.442691243336 * upper],  # P0_lv [MAP]
        [1.29130584142 * lower, 1.29130584142 * upper],  # P0_rv [MAP]
        [0.034856261421 * lower, 0.034856261421 * upper],  # s [MAP]
        [21.617810506166 * lower, 21.617810506166 * upper],  # fab_o [MAP]
        [17.175133514906 * lower, 17.175133514906 * upper],  # fes_o [MAP]
        [2.07231183456 * lower, 2.07231183456 * upper],  # fes_inf [MAP]
        [80 * lower, 80 * upper],
        [2.770365085494 * lower, 2.770365085494 * upper],  # fev_o [MAP]
        [7.423024627183 * lower, 7.423024627183 * upper],  # fev_inf [MAP]
        [0.05880583949 * lower, 0.05880583949 * upper],  # kes [MAP]
        [7.06 * lower, 7.06 * upper],
        [0.658 * lower, 0.658 * upper],
        [0.65 * lower, 0.65 * upper],
        [0.511820670232 * lower, 0.511820670232 * upper],  # Io_sv [MAP]
        [0.126 * lower, 0.126 * upper],
        [0.114 * lower, 0.114 * upper],
        [0.13 * lower, 0.13 * upper],
        [0.080130200394 * lower, 0.080130200394 * upper],  # kcc_sv [MAP]
        [0.0162 * lower, 0.0162 * upper],
        [9 * lower, 9 * upper],
        [-0.0283 * upper, -0.0283 * lower],
        [5.5 * lower, 5.5 * upper],
        [-0.037 * upper, -0.037 * lower],
        [64.9 * lower, 64.9 * upper],
        [-0.437 * upper, -0.437 * lower],
        [1.9 * lower, 1.9 * upper],
        [-0.0008 * upper, -0.0008 * lower],
        [-0.68 * upper, -0.68 * lower],
        [-2.018544472453 * upper, -2.018544472453 * lower],  # Wb_sh [MAP]
        [-1.1375 * upper, -1.1375 * lower],
        [-1.004859455505 * upper, -1.004859455505 * lower],  # Wb_sv [MAP]
        [1 * lower, 1 * upper],
        [1.716 * lower, 1.716 * upper],
        [1.716 * lower, 1.716 * upper],
        [0.2 * lower, 0.2 * upper],
        [-0.3997 * upper, -0.3997 * lower],
        [-0.3997 * upper, -0.3997 * lower],
        [-0.103 * upper, -0.103 * lower],
        [0.4 * lower, 0.4 * upper],
        [0.4 * lower, 0.4 * upper],
        [0.4 * lower, 0.4 * upper],
        [0.4 * lower, 0.4 * upper],
        [2.060317555315 * lower, 2.060317555315 * upper],  # Emax_lv0 [MAP]
        [1.279086612055 * lower, 1.279086612055 * upper],  # Emax_rv0 [MAP]
        [2.996981540095 * lower, 2.996981540095 * upper],  # fes_min [MAP]
        [0.431945385857 * lower, 0.431945385857 * upper],  # GEmax_lv [MAP]
        [0.282 * lower, 0.282 * upper],
        [2.47 * lower, 2.47 * upper],
        [1.94 * lower, 1.94 * upper],
        [2.47 * lower, 2.47 * upper],
        [0.695 * lower, 0.695 * upper],
        [-58.29 * upper, -58.29 * lower],
        [-74.21 * upper, -74.21 * lower],
        [-58.29 * upper, -58.29 * lower],
        [-237.648313155424 * upper, -237.648313155424 * lower],  # GV_sv [MAP]
        [3.51 * lower, 3.51 * upper],
        [1.655 * lower, 1.655 * upper],
        [5.27 * lower, 5.27 * upper],
        [2.49 * lower, 2.49 * upper],
        [1 * lower, 1 * upper],
        [1.5 * lower, 1.5 * upper],
        [6 * lower, 6 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [45 * lower, 45 * upper],
        [30 * lower, 30 * upper],
        [30 * lower, 30 * upper],
        [3.6 * lower, 3.6 * upper],
        [13.32 * lower, 13.32 * upper],
        [11.1940122515 * lower, 11.1940122515 * upper],  # theta_svn [MAP]
        [53 * lower, 53 * upper],
        [6 * lower, 6 * upper],
        [6 * lower, 6 * upper],
        [33.674334951783 * lower, 33.674334951783 * upper],  # PaCO2_n [MAP]
        [47.041027400563 * lower, 47.041027400563 * upper],  # f_ab_max [MAP]
        [2.52 * lower, 2.52 * upper],
        [10.038635495577 * lower, 10.038635495577 * upper],  # k_ab [MAP]
        [101.115183178836 * lower, 101.115183178836 * 1.15],  # P_n [MAP]
        [120 * 0.9, 120 * upper],
        [1.4 * lower, 1.4 * upper],
        [12.3 * lower, 12.3 * upper],
        [0.835 * lower, 0.835 * upper],
        [29.27 * lower, 29.27 * upper],
        [3 * lower, 3 * upper],
        [45 * lower, 45 * upper],
        [11.76 * lower, 11.76 * upper],
        [-0.112650592783 * upper, -0.112650592783 * lower],  # GT_s [MAP]
        [0.100153101641 * lower, 0.100153101641 * upper],  # GT_v [MAP]
        [0.654880149316 * lower, 0.654880149316 * upper],  # T0 [MAP]
        [20.9 * lower, 20.9 * upper],
        [92.8 * lower, 92.8 * upper],
        [10570 * lower, 10570 * upper],
        [-5.251 * upper, -5.251 * lower],
        [0.158362780305 * lower, 0.158362780305 * upper],  # Cvb_O2_n [MAP]
        [10 * lower, 10 * upper],
        [0.925 * lower, 0.925 * upper],
        [6.57 * lower, 6.57 * upper],
        [0.11 * lower, 0.11 * upper],
        [0.155 * lower, 0.155 * upper],
        [35 * lower, 35 * upper],
        [30 * lower, 30 * upper],
        [11.11 * lower, 11.11 * upper],
        [142.8 * lower, 142.8 * upper],
        [0.4 * lower, 0.4 * upper],
        [0.86 * lower, 0.86 * upper],
        [19.71 * lower, 19.71 * upper],
        [12660 * lower, 12660 * upper],
        [0.135937785465 * lower, 0.135937785465 * upper],  # Cvam_O2_n [MAP]
        [30 * lower, 30 * upper],
        [40 * lower, 40 * upper],
        [0.36721777927 * lower, 0.36721777927 * upper],  # Io_met [MAP]
        [0.204798869807 * lower, 0.204798869807 * upper],  # kmet [MAP]
        [0.516 * lower, 0.516 * upper],
        [20 * lower, 20 * upper],
        [-1.87 * upper, -1.87 * lower],
        [1000 * lower, 1000 * upper],
        [5000 * lower, 5000 * upper],
        [2 * lower, 2 * upper],
        [7 * lower, 7 * upper],
        [1.309 * lower, 1.309 * upper],
        [1200 * lower, 1200 * upper],
        [200 * lower, 200 * upper],
        [2 * lower, 2 * upper],
        [4.198293711872 * lower, 4.198293711872 * upper],  # Kv_mi [MAP]
        [1.309 * lower, 1.309 * upper],
        [2293.073313146301 * lower, 2293.073313146301 * upper],  # Kp_po [MAP]
        [2000 * lower, 2000 * upper],
        [2 * lower, 2 * upper],
        [8.115098186092 * lower, 8.115098186092 * upper],  # Kv_po [MAP]
        [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper],
        [200 * lower, 200 * upper],
        [2 * lower, 2 * upper],
        [2.958475913209 * lower, 2.958475913209 * upper],  # Kv_tr [MAP]
        [1.309 * lower, 1.309 * upper],
        [0.0000317 * lower, 0.0000317 * upper],
        [393.134478510141 * lower, 393.134478510141 * upper],  # R_po [MAP]
        [400 * lower, 400 * upper],
        [400 * lower, 400 * upper],
        [350 * lower, 350 * upper],
        [0.001513621371 * lower, 0.001513621371 * upper],  # C_O2_param1 [MAP]
        [2.891202516105 * lower, 2.891202516105 * upper],  # C_O2_param2 [MAP]
        [0.0000303 * lower, 0.0000303 * upper],
        [104 * lower, 104 * upper],
        [311.969906535 * lower, 311.969906535 * upper],  # Vu_bv [MAP]
        [93.16 * lower, 93.16 * upper],
        [513.955410368231 * lower, 513.955410368231 * upper],  # Vu_jp [MAP]
        [123 * lower, 123 * upper],
        [116.68 * lower, 116.68 * upper],
        [114 * lower, 114 * upper],
        [27.301439733132 * lower, 27.301439733132 * upper],  # Vu_la [MAP]
        [17.35873806827 * lower, 17.35873806827 * upper],  # Vu_lv [MAP]
        [32.331282965822 * lower, 32.331282965822 * upper],  # Vu_ra [MAP]
        [43.648498038143 * lower, 43.648498038143 * upper],  # Vu_rv [MAP]
        [8 * lower, 8 * upper],
        [8 * lower, 8 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [20 * lower, 20 * upper],
        [20 * lower, 20 * upper],
        [20 * lower, 20 * upper],
        [20 * lower, 20 * upper],
        [246.325150657968 * lower, 246.325150657968 * upper],  # Vu_amv0 [MAP]
        [646.987007375066 * lower, 646.987007375066 * upper],  # Vu_ev0 [MAP]
        [190.95 * lower, 190.95 * upper],
        [1262.699536885199 * lower, 1262.699536885199 * upper],  # Vu_sv0 [MAP]
        [20 * lower, 20 * upper],
        [30 * lower, 30 * upper],
        [2.076 * lower, 2.076 * upper],
        [0.8 * lower, 0.8 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [1.5 * lower, 1.5 * upper],
        [20 * lower, 20 * upper],
        [10 * lower, 10 * upper],
        [5 * lower, 5 * upper],
        [40 * lower, 40 * upper],
        [10 * lower, 10 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [5 * lower, 5 * upper],
        [5 * lower, 5 * upper],
        [5 * lower, 5 * upper],
        [5 * lower, 5 * upper],
        [2 * lower, 2 * upper],
        [0.2 * lower, 0.2 * upper],
        [4 * lower, 4 * upper],
        [0.3 * lower, 0.3 * upper],
        [0.012573867389 * lower, 0.012573867389 * upper],  # KE_lv [MAP]
        [0.00928516807 * lower, 0.00928516807 * upper],  # KE_rv [MAP]
        [0.1 * lower, 0.1 * upper],
        [0.2 * lower, 0.2 * upper],
        [3 * lower, 3 * upper],
        [2.5 * lower, 2.5 * upper],
        [20 * lower, 20 * upper],
        [0.01 * lower, 0.01 * upper],
        [50 * lower, 50 * upper],
        [0.25 * lower, 0.25 * upper],
        [0.25 * lower, 0.25 * upper],
        [50 * lower, 50 * upper],
        [4.300117727822 * lower, 4.300117727822 * upper],  # scale_param1 [MAP]
        [0.3 * lower, 0.3 * upper],
        [30.621117364076 * lower, 30.621117364076 * upper],  # scale_param4 [MAP]
        [0.04 * lower, 0.04 * upper],
        [80 * lower, 80 * upper],
        [0.038470374015 * lower, 0.038470374015 * upper],  # rise_time_atr [MAP]
        [0.351802683206 * lower, 0.351802683206 * upper],  # rise_time_ven [MAP]
        [0.504893619857 * 0.85, 0.504893619857 * 1.15],  # fall_time_ven [MAP]
        [0.941623732815 * 0.92, 0.941623732815 * 1.08],  # ahead1 [MAP]
        [0.0873 * lower, 0.0873 * upper],
        [1.101085481093 * 0.85, 1.101085481093 * 1.15],  # r [MAP]
        [1.356014731 * 0.85, 1.356014731 * 1.15],  # l [MAP]
        [132.141948246527 * lower, 132.141948246527 * upper],  # V_nominal [MAP]
        [50.289524292913 * lower, 50.289524292913 * upper],  # V_scale [MAP]
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



dgsm_summary = OrderedDict()

# min_frac = 0.003  # 0.3%
min_frac = 0.01
# min_frac = 0.0

for col in range(Result.shape[1]):

    Y = Result[:, col]
    output_label = output_names[col]

    # Skip invalid outputs
    if not np.all(np.isfinite(Y)):
        print(f"Skipping {output_label} (non-finite values)")
        continue

    # DGSM analysis
    Si = dgsm.analyze(sp, X, Y, print_to_console=False)

    dgsm_vals = np.array(Si["dgsm"], dtype=float)
    names = np.array(Si["names"])

    # Sort descending
    order = np.argsort(dgsm_vals)[::-1]
    dgsm_sorted = dgsm_vals[order]
    names_sorted = names[order]

    total = dgsm_sorted.sum()
    if total <= 0 or not np.isfinite(total):
        print(f"Skipping {output_label} (non-positive/invalid DGSM total: {total})")
        continue

    # --- NEW: keep only params contributing >= 0.3% of total ---
    thresh = min_frac * total
    keep = dgsm_sorted >= thresh
    dgsm_kept = dgsm_sorted[keep]
    names_kept = names_sorted[keep]

    # Cumulative sensitivity over kept params, but target is 90% of ORIGINAL total
    cumu = np.cumsum(dgsm_kept)

    target = 0.9 * total
    if cumu.size == 0:
        idx_90 = 0
        top_names_90 = np.array([])
        top_dgsm_90 = np.array([])
        reached = 0.0
    else:
        idx_90 = np.searchsorted(cumu, target) + 1
        # if you can't reach 90% without including <0.3% params, cap at all kept
        idx_90 = min(idx_90, len(cumu))
        top_names_90 = names_kept[:idx_90]
        top_dgsm_90 = dgsm_kept[:idx_90]
        reached = cumu[idx_90 - 1] / total

    dgsm_summary[output_label] = {
        "n_params_90": idx_90,
        "param_names": top_names_90,
        "dgsm_values": top_dgsm_90,
        "min_frac": min_frac,
        "fraction_of_total_reached": reached,
        "n_params_passing_0p3pct": int(keep.sum()),
    }

    # ---- Console output ----
    print("\n" + "=" * 80)
    print(f"Output: {output_label}")
    print(f"Min per-parameter contribution: {min_frac*100:.1f}% (DGSM >= {thresh:.4e})")
    print(f"Parameters selected (up to 90% total, with cutoff): {idx_90}")
    print(f"Fraction of total DGSM reached: {reached*100:.2f}%")
    if reached < 0.90:
        print("NOTE: Could not reach 90% without including parameters < 1.0% each.")

    print("-" * 80)
    for name, val in zip(top_names_90, top_dgsm_90):
        print(f"{name:25s} : {val:.4e}  ({val/total*100:.3f}%)")


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




