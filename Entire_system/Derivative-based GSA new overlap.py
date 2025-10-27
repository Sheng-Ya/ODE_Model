import numpy as np
import matplotlib.pyplot as plt
import dgsm_edited as dgsm
from SALib import ProblemSpec

# X = np.load('All_params_DGSM_500_X_samples_rest_20.npy')
# Result1 = np.load('All_params_DGSM_500_Result_rest_1_250_20.npy')
# Result2 = np.load('All_params_DGSM_500_Result_rest_250_500_20.npy')
# Result = np.vstack((Result1, Result2))

# X = np.load('All_params_DGSM_500_X_samples_rest_20.npy')
# Result1 = np.load('All_params_DGSM_500_Result_rest_0_250_20_exercise_backup.npy')
# Result2 = np.load('All_params_DGSM_500_Result_exercise_250_500_backup.npy')
# Result = np.vstack((Result1, Result2))

X = np.load('All_params_DGSM_500_X_samples_rest_50.npy')
Result1 = np.load('All_params_DGSM_500_Result_rest_1_250_50.npy')
Result2 = np.load('All_params_DGSM_500_Result_rest_250_500_50.npy')
Result = np.vstack((Result1, Result2))

lower = 0.5
upper = 1.5

Stroke_Volume = Result[:, 3] - Result[:, 4]
Ejection_fraction = (Stroke_Volume / Result[:, 3]) * 100
Result = np.column_stack((Result, Stroke_Volume))
Result = np.column_stack((Result, Ejection_fraction))

D = 300
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
mask_blocks = mask_blocks & mask_blocks_nan
# Expand mask to all rows in a block
mask_full = np.repeat(mask_blocks, block_size)
# Filter arrays
X = X[mask_full]
Result = Result[mask_full]



# Adjust this to match your problem spec
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
        # gas
        [0.03255 * 0.9, 0.03255 * 1.1], [87 * 0.9, 87 * 1.1],
        [194.4 * 0.9, 194.4 * 1.1], [1.819 * 0.9, 1.819 * 1.1],
        [0.05591 * 0.9, 0.05591 * 1.1], [0.015 * lower, 0.015 * upper],
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
        [0.15 * lower, 0.15 * upper], [0.3 * lower, 0.3 * upper], [0.9 * 0.95, 0.9 * 1.05],
        [0.0872665 * lower, 0.0872665 * upper], [0.3 * lower, 0.3 * upper]]
})

# Plot
Result_cols = ["Heart Rate", "Systolic Pressure", "Diastolic Pressure", "EDV", "ESV", "Max RV Volume", "Min RV Volume",
               "Max RV Pressure", "Min RV Pressure", "Min RA Volume", "Max RA Volume", "Min RA Pressure", "Max RA Pressure",
               "Min LA Volume", "Max LA Volume", "Min LA Pressure", "Max LA Pressure", "LA EDV", "RA EDV", "LV Max P Deriv",
               "RV Max P Deriv", "Max Tidal Volume", "Minute Ventilation", "Cardiac Output", "PaO2", "PaCO2",
               "Stroke Volume", "Ejection Fraction"]

n_biomarkers = len(Result_cols)  # i = 0 ... 10

# Dictionary mapping biomarker index -> influential parameters
influential_params = {}

for i in range(n_biomarkers):
    biomarker = Result[:, i]
    Si = dgsm.analyze(sp, X, biomarker, print_to_console=False)

    dgsm1 = np.array(Si['dgsm'])
    names = np.array(Si['names'])
    conf = np.array(Si['dgsm_conf'])

    # Sort descending
    dgsm_sorted = np.argsort(dgsm1)[::-1]
    top_dgsm = dgsm1[dgsm_sorted]
    top_names = names[dgsm_sorted]
    top_conf = conf[dgsm_sorted]

    # Cumulative contribution
    cumusum = np.cumsum(top_dgsm)
    total = cumusum[-1]

    threshold_index = np.searchsorted(cumusum, 0.90 * total) + 1
    vars_90 = top_names[:threshold_index]

    influential_params[i] = set(vars_90)

    print(f"{Result_cols[i]}: {threshold_index} parameters contribute 90% sensitivity")
    print(vars_90)
    print()

# Union of all influential parameters across biomarkers
union_params = set().union(*influential_params.values())
print(f"Union of parameters (all biomarkers): {union_params}")

# --- Plot which biomarkers each parameter is influential in ---
param_counts = {p: sum(p in influential_params[i] for i in range(n_biomarkers))
                for p in union_params}

# Sort parameters by count descending
param_list = sorted(union_params, key=lambda p: param_counts[p], reverse=True)

# Rebuild matrix
matrix = np.zeros((len(param_list), n_biomarkers), dtype=int)
for j, p in enumerate(param_list):
    for i in range(n_biomarkers):
        if p in influential_params[i]:
            matrix[j, i] = 1


# Count how many parameters influence each biomarker (for sorting y-axis)
biomarker_counts = np.sum(matrix, axis=0)  # sum over rows for each column
sorted_biomarkers_idx = np.argsort(biomarker_counts)[::-1]

# Count how many biomarkers each parameter influences (for sorting x-axis)
param_counts = np.sum(matrix, axis=1)
sorted_params_idx = np.argsort(param_counts)[::-1]

# Reorder matrix and labels
matrix_sorted = matrix[np.ix_(sorted_params_idx, sorted_biomarkers_idx)]
param_list_sorted = [param_list[i] for i in sorted_params_idx]
Result_cols_sorted = [Result_cols[i] for i in sorted_biomarkers_idx]

print(len(param_list_sorted))

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(matrix_sorted.T, aspect="auto", cmap="Blues")  # transpose to switch axes

# Set ticks
ax.set_xticks(range(len(param_list_sorted)))
ax.set_xticklabels(param_list_sorted, rotation=90)  # parameters on x-axis
ax.set_yticks(range(n_biomarkers))
ax.set_yticklabels(Result_cols_sorted)  # biomarkers on y-axis

# plt.colorbar(im, ax=ax, label="Influential (1 = yes, 0 = no)")
plt.title("Influential Parameters (contributing to 90% DGSM) across Biomarkers")
plt.tight_layout()
plt.show()