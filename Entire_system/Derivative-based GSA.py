from SALib import ProblemSpec
from SALib.plotting.bar import plot as barplot
# from SALib.analyze import dgsm
import dgsm_edited as dgsm
import matplotlib.pyplot as plt
import numpy as np

X = np.load('New_DGSM_500_X_samples_HR_P_sys_P_dia_no_bifur_delay.npy')[:83172, :]

# Result_2 =  np.array([[0.0,0.0,0.0]])
# Result = np.load('Result_DGSM_new.npy')[:1800]
Result = np.load('Result_DGSM_478_delay.npy')
# Result1 = np.load('DGSM_500_Result_HR_P_sys_P_dia_steady_remove_120s.npy')


#
# Result_3 = np.concatenate((Result, Result_2))
# np.save("Results_first_41375_chunk.npy", Result_3)

HR = Result[:, 0]
# HR = HR_load[HR_load != 0]

# # Assume your arrays are named
# first_array = np.load('DGSM_500_X_samples_HR_P_sys_P_dia_steady_remove.npy')   # shape (N, ...)
# second_array = np.load('DGSM_500_Result_HR_P_sys_P_dia_steady_remove_120s.npy')[:,0]  # shape (N,)
#
# Total number of blocks
# block_size = 180
# num_blocks = len(second_array) // block_size
#
# # Indices to keep
# indices_to_keep = np.ones(len(second_array), dtype=bool)
#
# for i in range(num_blocks):
#     base_idx = i * block_size
#     if second_array[base_idx] == 0:
#         indices_to_keep[base_idx:base_idx + block_size] = False
#
# # Apply filtering
# second_array_filtered = second_array[indices_to_keep]
#
# np.save("DGSM_500_X_samples_HR_P_sys_P_dia_filtered.npy", first_array_filtered)









lower = 0.8
upper = 1.2

sp = ProblemSpec({
    'outputs': ["HR"],

    'names': [
        "beta1", "beta2", "C2", "K1", "K2", "a2", "alpha1", "alpha2", "dc", "KCCO2",
        # "MRBCO2",
        "GV_dead",
        # "Kbg",
        "KcCO2", "KcMRV", "KpCO2", "KpO2", "V0_dead", "VA_rest", "Pmax",
        "Pmax_dot", "E_rs", "R_rs",
        "C_sa", "L_sa", "R_sa", "C_amv", "C_bv",
        "C_ev", "C_hv", "C_rmv", "C_sv", "R_amv_n", "R_bv_n",
        "R_ev_n", "R_hv_n", "R_rmv_n", "R_sv_n", "D1", "D2", "K1_vc", "K2_vc", "Kr_vc", "Rvc_n",
        "C_pa", "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv", "Emax_la", "P0_la", "Emax_ra",
        "P0_ra", "P0_lv", "P0_rv", "g_abd", "g_thor", "P_abdmax_n", "P_abdmin_n",
        # "P_thormax_n", "P_thormin_n",
        "VT_n", "A_im", "Tc", "T_im", "s",
        # cardio control
        "fab_o", "fes_o", "fes_inf", "fes_max", "fev_o", "fev_inf",
        "kes", "kev", "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v", "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min",
        "Ysv_max", "Ysv_min", "Yv_max", "Yv_min", "theta_v", "Wb_sh", "Wb_sp", "Wb_sv", "Wc_sh", "Wc_sp",
        "Wc_sv", "Wc_v", "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
        "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
        "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "AT", "g_ccsh", "g_ccsp",
        "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv", "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn", "theta_spn",
        "theta_svn", "x_sh", "x_sp", "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min", "k_ab", "P_n", "f_acCO2_n", "f_ac_max",
        "f_ac_min", "k_ac", "K_H", "PaO2_ac_n", "G_ap", "GT_s", "GT_v", "T0", "A", "B",
        "C", "D", "Cvb_O2_n", "gb_O2", "MO2_bp", "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2", "grm_O2",
        "Kh_CO2", "Krm_CO2", "MO2_hpn", "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n", "gam_O2", "gM", "kmet",
        "MO2_ampn", "phi_max", "phi_min"
    ],

    'bounds': [
        # gas
        [0.008275 * lower, 0.008275 * upper], [0.03255 * lower, 0.03255 * upper], [40 * lower, 40 * upper],
        [13 * lower, 13 * upper], [25 * lower, 25 * upper], [1.219 * lower, 1.219 * upper],
        [0.03198 * lower, 0.03198 * upper], [0.05591 * lower, 0.05591 * upper], [0.015 * lower, 0.015 * upper],
        [346000 * lower, 346000 * upper],
        # [0.0009 * lower, 0.0009 * upper],
        # resp control
        [0.1698 * lower, 0.1698 * upper],
        # [17.4 * lower, 17.4 * upper],
        [0.2332 * lower, 0.2332 * upper],
        [1 * lower, 1 * upper], [0.2025 * lower, 0.2025 * upper], [4.72e-09 * lower, 4.72e-09 * upper],
        [0.1587 * lower, 0.1587 * upper], [0.067 * lower, 0.067 * upper], [50 * lower, 50 * upper],
        [1000 * lower, 1000 * upper], [21.9 * lower, 21.9 * upper], [3.02 * lower, 3.02 * upper],
        # cardio
        [0.28 * lower, 0.28 * upper], [0.00066 * lower, 0.00066 * upper], [0.2 * lower, 0.2 * upper],
        [9.4 * lower, 9.4 * upper],
        [10.71 * lower, 10.71 * upper], [20 * lower, 20 * upper],
        [3.57 * lower, 3.57 * upper],
        [6.28 * lower, 6.28 * upper], [61.11 * lower, 61.11 * upper],
        [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper],
        [0.224 * lower, 0.224 * upper], [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper],
        [0.3855 * lower, 0.3855 * upper], [-5 * upper, -5 * lower], [0.15 * lower, 0.15 * upper],
        [0.4 * lower, 0.4 * upper], [0.001 * lower, 0.001 * upper], [0.075 * lower, 0.075 * upper],
        [0.76 * lower, 0.76 * upper], [5.8 * lower, 5.8 * upper], [20.5 * lower, 20.5 * upper],
        [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
        [0.06 * lower, 0.06 * upper], [0.25 * lower, 0.25 * upper], [0.55 * lower, 0.55 * upper],
        [0.25 * lower, 0.25 * upper], [0.55 * lower, 0.55 * upper], [1.5 * lower, 1.5 * upper],
        [1.5 * lower, 1.5 * upper], [3.39 * lower, 3.39 * upper], [6.8 * lower, 6.8 * upper],
        [-1 * upper, -1 * lower], [-2.5 * upper, -2.5 * lower],
        # [-1 * upper, -1 * lower],
        # [-2 * upper, -2 * lower],
        [0.45 * lower, 0.45 * upper], [50 * lower, 50 * upper],
        [0.7 * lower, 0.7 * upper], [1.1 * lower, 1.1 * upper], [0.04 * lower, 0.04 * upper],
        # cardio control
        [25 * lower, 25 * upper], [16.11 * lower, 16.11 * upper], [2.1 * lower, 2.1 * upper],
        [80 * lower, 80 * upper], [3.2 * lower, 3.2 * upper], [6.3 * lower, 6.3 * upper],
        [0.0675 * lower, 0.0675 * upper], [7.06 * lower, 7.06 * upper], [0.114 * lower, 0.114 * upper],
        [0.13 * lower, 0.13 * upper], [0.09 * lower, 0.09 * upper], [0.0162 * lower, 0.0162 * upper],
        [9 * lower, 9 * upper], [-0.0283 * upper, -0.0283 * lower], [5.5 * lower, 5.5 * upper],
        [-0.037 * upper, -0.037 * lower], [64.9 * lower, 64.9 * upper], [-0.028 * upper, -0.028 * lower],
        [1.9 * lower, 1.9 * upper], [-0.0008 * upper, -0.0008 * lower], [-0.68 * upper, -0.68 * lower],
        [-1.75 * upper, -1.75 * lower], [-1.1375 * upper, -1.1375 * lower], [-1.1375 * upper, -1.1375 * lower],
        [1 * lower, 1 * upper], [1.716 * lower, 1.716 * upper], [1.716 * lower, 1.716 * upper],
        [0.2 * lower, 0.2 * upper], [-0.2 * upper, -0.2 * lower], [-0.3997 * upper, -0.3997 * lower],
        [-0.3997 * upper, -0.3997 * lower], [-0.103 * upper, -0.103 * lower], [0.4 * lower, 0.4 * upper],
        [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper],
        [1.412 * lower, 1.412 * upper], [0.7 * lower, 0.7 * upper], [2.66 * lower, 2.66 * upper],
        [0.475 * lower, 0.475 * upper], [0.282 * lower, 0.282 * upper], [2.47 * lower, 2.47 * upper],
        [1.94 * lower, 1.94 * upper], [2.47 * lower, 2.47 * upper], [0.695 * lower, 0.695 * upper],
        [-58.29 * upper, -58.29 * lower], [-74.21 * upper, -74.21 * lower], [-58.29 * upper, -58.29 * lower],
        [-265.4 * upper, -265.4 * lower], [3.51 * lower, 3.51 * upper], [1.655 * lower, 1.655 * upper],
        [5.27 * lower, 5.27 * upper], [2.49 * lower, 2.49 * upper], [(1 / 60) * lower, (1 / 60) * upper],
        [1 * lower, 1 * upper], [1.5 * lower, 1.5 * upper], [0.2 * lower, 0.2 * upper],
        [6 * lower, 6 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
        [45 * lower, 45 * upper], [30 * lower, 30 * upper], [30 * lower, 30 * upper],
        [3.6 * lower, 3.6 * upper], [13.32 * lower, 13.32 * upper], [13.32 * lower, 13.32 * upper],
        [53 * lower, 53 * upper], [6 * lower, 6 * upper], [6 * lower, 6 * upper],
        [40 * lower, 40 * upper], [47.78 * lower, 47.78 * upper], [2.52 * lower, 2.52 * upper],
        [11.76 * lower, 11.76 * upper], [92 * lower, 92 * upper], [1.4 * lower, 1.4 * upper],
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
        [0.18 * lower, 0.18 * upper], [0.516 * lower, 0.516 * upper], [20 * lower, 20 * upper],
        [-1.87 * upper, -1.87 * lower],
    ],
})

# HR = Result[:, 0]
# HR_sorted_index = np.argsort(HR)[::-1]
# HR_sorted = HR[HR_sorted_index]
#
# fig, ax1 = plt.subplots()
# ax1.bar(range(600), HR_sorted[500:1100], label="HR", color="b")
#
# plt.show()

Si = dgsm.analyze(sp, X, HR, print_to_console=True)

## Extract and sort
dgsm = np.array(Si['dgsm'])
names = np.array(Si['names'])
conf = np.array(Si['dgsm_conf'])

dgsm_sorted = np.argsort(dgsm)[::-1]  # descending order
top_dgsm = dgsm[dgsm_sorted]
top_names = names[dgsm_sorted]
top_conf = conf[dgsm_sorted]

# top_dgsm = top_dgsm[7:]
# top_names = top_names[7:]

# Split the sorted arrays in half
mid = len(top_dgsm) // 2
dgsm_1, dgsm_2 = top_dgsm[:mid], top_dgsm[mid:]
names_1, names_2 = top_names[:mid], top_names[mid:]
conf_1, conf_2 = top_conf[:mid], top_conf[mid:]

# Plot first half
plt.figure(figsize=(10, 8))
plt.bar(names_1, dgsm_1, yerr=conf_1)
plt.xlabel("Sensitivity Index (DGSM)")
plt.title("HR DGSM 147")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Plot second half
plt.figure(figsize=(10, 8))
plt.bar(names_2, dgsm_2, yerr=conf_2)
plt.xlabel("Sensitivity Index (DGSM)")
plt.title("HR DGSM 147")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()




# Calculate cumulative sum and total sum
cumusum = np.cumsum(top_dgsm)
total = cumusum[-1]

# Find the index where cumulative sum reaches 95% of total
threshold_index = np.searchsorted(cumusum, 0.95 * total) + 1  # +1 to include that index

# Get variables contributing to 95% of sensitivity
vars_95 = top_names[:threshold_index]
sens_95 = top_dgsm[:threshold_index]

print(f"Number of variables contributing 95% sensitivity: {threshold_index}")
print("Variables:")
for var, sens in zip(vars_95, sens_95):
    print(f"{var}: {sens}")

# Optional: Plot these variables only
plt.figure(figsize=(10, 6))
plt.bar(vars_95, sens_95)
plt.xlabel("Parameters")
plt.ylabel("DGSM Sensitivity")
plt.title("Parameters contributing 95% of DGSM Sensitivity")
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()