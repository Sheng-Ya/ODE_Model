from SALib import ProblemSpec
from SALib.plotting.bar import plot as barplot
# from SALib.analyze import dgsm
import dgsm_edited as dgsm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

X = np.load('DGSM_250_X_samples_HR_P_sys_P_dia_steady_remove.npy')
Result_load = np.load('DGSM_250_Result_HR_P_sys_P_dia_steady_remove.npy')

Result = np.insert(Result_load, 41374, [[0, 0, 0]], axis=0)

HR = Result[:, 0]

# # Only use failed points
# X_failed = X[HR == 0]
# X_failed_2d = TSNE(n_components=2, random_state=42).fit_transform(X_failed)
#
# plt.figure(figsize=(8, 6))
# plt.scatter(X_failed_2d[:, 0], X_failed_2d[:, 1], c='red', alpha=0.6, s=10)
# plt.title("t-SNE of Failed Simulations Only")
# plt.xlabel("t-SNE Dim 1")
# plt.ylabel("t-SNE Dim 2")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Assume X is (18400, 183), Result is (18400,)
# success_mask = HR != 0
# fail_mask = HR == 0
#
# # Run t-SNE
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# X_2d = tsne.fit_transform(X)
#
# # Plot
# plt.figure(figsize=(10, 6))
# plt.scatter(X_2d[success_mask, 0], X_2d[success_mask, 1], c='green', label='Success', alpha=0.5, s=10)
# plt.scatter(X_2d[fail_mask, 0], X_2d[fail_mask, 1], c='red', label='Failure', alpha=0.5, s=10)
# plt.legend()
# plt.title("t-SNE Projection of Simulation Inputs")
# plt.xlabel("Dimension 1")
# plt.ylabel("Dimension 2")
# plt.grid(True, linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()













lower = 0.8
upper = 1.2

sp = ProblemSpec({
    'outputs': ["HR"],

    'names': [
        "beta1", "beta2", "C2", "K1", "K2", "a2", "alpha1", "alpha2", "dc", "KCCO2",
        "MRBCO2", "GV_dead", "Kbg", "KcCO2", "KcMRV", "KpCO2", "KpO2", "V0_dead", "VA_rest", "Pmax",
        "Pmax_dot", "E_rs", "R_rs", "C_sa", "L_sa", "R_sa", "C_amp", "C_amv", "C_bp", "C_bv",
        "C_ep", "C_ev", "C_hp", "C_hv", "C_rmp", "C_rmv", "C_sp", "C_sv", "R_amv_n", "R_bv_n",
        "R_ev_n", "R_hv_n", "R_rmv_n", "R_sv_n", "D1", "D2", "K1_vc", "K2_vc", "Kr_vc", "Rvc_n",
        "C_pa", "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv", "Emax_la", "P0_la", "Emax_ra",
        "P0_ra", "P0_lv", "P0_rv", "g_abd", "g_thor", "P_abdmax_n", "P_abdmin_n", "P_thormax_n", "P_thormin_n", "VT_n",
        "A_im", "Tc", "T_im", "s", "fab_o", "fes_o", "fes_inf", "fes_max", "fev_o", "fev_inf",
        "kes", "kev", "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v", "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min",
        "Ysv_max", "Ysv_min", "Yv_max", "Yv_min", "theta_v", "Wb_sh", "Wb_sp", "Wb_sv", "Wc_sh", "Wc_sp",
        "Wc_sv", "Wc_v", "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
        "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
        "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "AT", "g_ccsh", "g_ccsp",
        "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv", "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn", "theta_spn", "theta_svn",
        "x_sh", "x_sp", "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min", "k_ab", "P_n", "f_acCO2_n", "f_ac_max",
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
        [346000 * lower, 346000 * upper], [0.0009 * lower, 0.0009 * upper],
        # resp control
        [0.1698 * lower, 0.1698 * upper], [17.4 * lower, 17.4 * upper], [0.2332 * lower, 0.2332 * upper],
        [1 * lower, 1 * upper], [0.2025 * lower, 0.2025 * upper], [4.72e-09 * lower, 4.72e-09 * upper],
        [0.1587 * lower, 0.1587 * upper], [0.067 * lower, 0.067 * upper], [50 * lower, 50 * upper],
        [1000 * lower, 1000 * upper], [21.9 * lower, 21.9 * upper], [3.02 * lower, 3.02 * upper],
        # cardio
        [0.28 * lower, 0.28 * upper], [0.00022 * lower, 0.00022 * upper], [0.06 * lower, 0.06 * upper],
        [0.315 * lower, 0.315 * upper], [9.4 * lower, 9.4 * upper], [0.358 * lower, 0.358 * upper],
        [10.71 * lower, 10.71 * upper], [0.668 * lower, 0.668 * upper], [20 * lower, 20 * upper],
        [0.119 * lower, 0.119 * upper], [3.57 * lower, 3.57 * upper], [0.21 * lower, 0.21 * upper],
        [6.28 * lower, 6.28 * upper], [2.05 * lower, 2.05 * upper], [61.11 * lower, 61.11 * upper],
        [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper],
        [0.224 * lower, 0.224 * upper], [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper],
        [0.3855 * lower, 0.3855 * upper], [-5 * upper, -5 * lower], [0.15 * lower, 0.15 * upper],
        [0.4 * lower, 0.4 * upper], [0.001 * lower, 0.001 * upper], [0.0025 * lower, 0.0025 * upper],
        [8 * lower, 8 * upper], [10 * lower, 10 * upper], [25.37 * lower, 25.37 * upper],
        [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
        [0.0056 * lower, 0.0056 * upper], [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper],
        [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper], [1.5 * lower, 1.5 * upper],
        [1.5 * lower, 1.5 * upper], [3.39 * lower, 3.39 * upper], [6.8 * lower, 6.8 * upper],
        [-1 * upper, 0 * lower], [-2.5 * upper, -2.5 * lower], [-1 * upper, 0.0 * lower],
        [-3 * upper, 0.0 * lower], [0.45 * lower, 0.45 * upper], [50 * lower, 50 * upper],
        [0.75 * lower, 0.75 * upper], [1 * lower, 1 * upper], [0.04 * lower, 0.04 * upper],
        # cardio control
        [25 * lower, 25 * upper],    [16.11 * lower, 16.11 * upper],
        [2.1 * lower, 2.1 * upper],    [80 * lower, 80 * upper],
        [3.2 * lower, 3.2 * upper],    [6.3 * lower, 6.3 * upper],
        [0.0675 * lower, 0.0675 * upper],    [7.06 * lower, 7.06 * upper],
        [0.114 * lower, 0.114 * upper],    [0.13 * lower, 0.13 * upper],
        [0.09 * lower, 0.09 * upper],    [0.0162 * lower, 0.0162 * upper],
        [9 * lower, 9 * upper],    [-0.0283 * lower, -0.0283 * upper],
        [5.5 * lower, 5.5 * upper],    [-0.037 * upper, -0.037 * lower],
        [64.9 * lower, 64.9 * upper],    [-0.028 * upper, -0.028 * lower],
        [1.9 * lower, 1.9 * upper],    [-0.0008 * upper, -0.0008 * lower],
        [-0.68 * upper, -0.68 * lower],    [-1.75 * upper, -1.75 * lower],
        [-1.1375 * upper, -1.1375 * lower],    [-1.1375 * upper, -1.1375 * lower],
        [1 * lower, 1 * upper],    [1.716 * lower, 1.716 * upper],
        [1.716 * lower, 1.716 * upper],    [0.2 * lower, 0.2 * upper],
        [0 * lower, 0.1 * upper],    [-0.3997 * upper, -0.3997 * lower],
        [-0.3997 * upper, -0.3997 * lower],    [-0.103 * upper, -0.103 * lower],
        [0.4 * lower, 0.4 * upper],    [0.4 * lower, 0.4 * upper],
        [0.4 * lower, 0.4 * upper],    [0.4 * lower, 0.4 * upper],
        [2.392 * lower, 2.392 * upper],    [1.412 * lower, 1.412 * upper],
        [2.66 * lower, 2.66 * upper],    [0.475 * lower, 0.475 * upper],
        [0.282 * lower, 0.282 * upper],    [2.47 * lower, 2.47 * upper],
        [1.94 * lower, 1.94 * upper],    [2.47 * lower, 2.47 * upper],
        [0.695 * lower, 0.695 * upper],    [-58.29 * upper, -58.29 * lower],
        [-74.21 * upper, -74.21 * lower],    [-58.29 * upper, -58.29 * lower],
        [-265.4 * upper, -265.4 * lower],    [3.51 * lower, 3.51 * upper],
        [1.655 * lower, 1.655 * upper],    [5.27 * lower, 5.27 * upper],
        [2.49 * lower, 2.49 * upper],    [(1/60) * lower, (1/60) * upper],
        [1 * lower, 1 * upper],    [1.5 * lower, 1.5 * upper],
        [0 * lower, 0.1 * upper],    [6 * lower, 6 * upper],
        [2 * lower, 2 * upper],    [2 * lower, 2 * upper],
        [45 * lower, 45 * upper],    [30 * lower, 30 * upper],
        [30 * lower, 30 * upper],    [3.6 * lower, 3.6 * upper],
        [13.32 * lower, 13.32 * upper],    [13.32 * lower, 13.32 * upper],
        [53 * lower, 53 * upper],    [6 * lower, 6 * upper],
        [6 * lower, 6 * upper],    [40 * lower, 40 * upper],
        [47.78 * lower, 47.78 * upper],    [2.52 * lower, 2.52 * upper],
        [11.76 * lower, 11.76 * upper],    [92 * lower, 92 * upper],
        [1.4 * lower, 1.4 * upper],    [12.3 * lower, 12.3 * upper],
        [0.835 * lower, 0.835 * upper],    [29.27 * lower, 29.27 * upper],
        [3 * lower, 3 * upper],    [45 * lower, 45 * upper],
        [11.76 * lower, 11.76 * upper],    [-0.13 * upper, -0.13 * lower],
        [0.09 * lower, 0.09 * upper],    [0.58 * lower, 0.58 * upper],
        [20.9 * lower, 20.9 * upper],    [92.8 * lower, 92.8 * upper],
        [10570 * lower, 10570 * upper],    [-5.251 * upper, -5.251 * lower],
        [0.14 * lower, 0.14 * upper],    [10 * lower, 10 * upper],
        [0.925 * lower, 0.925 * upper],    [6.57 * lower, 6.57 * upper],
        [0.11 * lower, 0.11 * upper],    [0.155 * lower, 0.155 * upper],
        [35 * lower, 35 * upper],    [30 * lower, 30 * upper],
        [11.11 * lower, 11.11 * upper],    [142.8 * lower, 142.8 * upper],
        [0.4 * lower, 0.4 * upper],    [0.86 * lower, 0.86 * upper],
        [19.71 * lower, 19.71 * upper],    [12660 * lower, 12660 * upper],
        [0.1555 * lower, 0.1555 * upper],    [30 * lower, 30 * upper],
        [40 * lower, 40 * upper],    [0.18 * lower, 0.18 * upper],
        [0.516 * lower, 0.516 * upper],    [20 * lower, 20 * upper],
        [-1.87 * upper, -1.87 * lower],
    ],
})


failed_mask = HR == 0  # Boolean mask for failed runs
X_failed = X[failed_mask]  # Parameter sets that caused failure

step = 183 + 1  # one base + 183 perturbations = 184
Result_base = HR[0::step]      # shape: (100,)
X_base = X[0::step, :]             # shape: (100, 183)

print(np.sum(Result_base == 0))

failed_perturb_values = []
failed_param_indices = []

for i in range(len(Result_base)):  # 100 blocks
    base_result = Result_base[i]
    base_x = X_base[i]

    if base_result == 0:
        continue  # Skip entire block if base point failed

    # Get 183 perturbations for this base
    start = i * step + 1
    end = start + 183
    block_results = HR[start:end]
    block_X = X[start:end, :]

    for j, (r, x_row) in enumerate(zip(block_results, block_X)):
        if r == 0:
            # Compare to base_x to find which param was perturbed
            diff = np.where(x_row != base_x)[0]
            if len(diff) == 1:  # Only one parameter changed
                param_idx = diff[0]
                param_value = x_row[param_idx]

                failed_perturb_values.append(param_value)
                failed_param_indices.append(param_idx)


failed_perturb_values = np.array(failed_perturb_values)
failed_param_indices = np.array(failed_param_indices)



bounds = np.array(sp["bounds"])  # shape (183, 2)
lower_bounds = bounds[:, 0]
upper_bounds = bounds[:, 1]


# Map to normalized value
failed_normalized_values = (failed_perturb_values - lower_bounds[failed_param_indices]) / (upper_bounds[failed_param_indices] - lower_bounds[failed_param_indices])

# Get list of unique failed parameter indices
unique_param_ids = np.unique(failed_param_indices)
param_labels = [sp["names"][i] for i in unique_param_ids]

# plt.figure(figsize=(max(10, len(unique_param_ids) * 0.4), 6))
#
# for i, param_id in enumerate(unique_param_ids):
#     # Extract failed values for this parameter
#     values = failed_normalized_values[failed_param_indices == param_id]
#
#     # Plot them as dots (scatter) with jitter in x to avoid overlap
#     x_jitter = np.random.normal(loc=i, scale=0.05, size=len(values))  # slight x jitter
#     plt.scatter(x_jitter, values, alpha=0.7, s=20)
#
# # Add vertical lines between parameters
# for i in range(len(unique_param_ids) + 1):
#     plt.axvline(i - 0.5, color='lightgray', linestyle='--', linewidth=0.7)
#
# # Axes settings
# plt.xticks(ticks=np.arange(len(unique_param_ids)), labels=param_labels, rotation=90)
# plt.ylabel("Normalized Parameter Value")
# plt.title("Failed Simulations by Parameter and Normalized Value")
# plt.ylim(0, 1)
# plt.grid(axis='y', linestyle='--', alpha=0.4)
# plt.tight_layout()
# plt.show()




# base_point_values = []  # new list to store normalized base values
#
# for i in range(len(Result_base)):
#     base_result = Result_base[i]
#     base_x = X_base[i]
#
#     if base_result == 0:
#         continue
#
#     start = i * step + 1
#     end = start + 183
#     block_results = HR[start:end]
#     block_X = X[start:end, :]
#
#     for j, (r, x_row) in enumerate(zip(block_results, block_X)):
#         if r == 0:
#             diff = np.where(x_row != base_x)[0]
#             if len(diff) == 1:
#                 param_idx = diff[0]
#                 param_value = x_row[param_idx]
#
#                 failed_perturb_values.append(param_value)
#                 failed_param_indices.append(param_idx)
#
#                 # Store normalized base value for that parameter
#                 base_val = base_x[param_idx]
#                 norm_base_val = (base_val - lower_bounds[param_idx]) / (upper_bounds[param_idx] - lower_bounds[param_idx])
#                 base_point_values.append(norm_base_val)
#
#
# failed_perturb_values = np.array(failed_perturb_values)
# failed_param_indices = np.array(failed_param_indices)
# base_point_values = np.array(base_point_values)





# plt.figure(figsize=(max(10, len(unique_param_ids) * 0.4), 6))
#
# for i, param_id in enumerate(unique_param_ids):
#     values = failed_normalized_values[failed_param_indices == param_id]
#     base_vals = base_point_values[failed_param_indices == param_id]
#
#     # Jitter for failed points
#     x_jitter = np.random.normal(loc=i, scale=0.05, size=len(values))
#     plt.scatter(x_jitter, values, alpha=0.7, s=20, color='tab:red', label='Failed value' if i == 0 else None)
#
#     # Plot base point as a horizontal marker (no jitter)
#     plt.scatter([i] * len(base_vals), base_vals, alpha=0.7, s=40, color='tab:blue', marker='x', label='Base point' if i == 0 else None)
#
# # Add vertical lines between parameters
# for i in range(len(unique_param_ids) + 1):
#     plt.axvline(i - 0.5, color='lightgray', linestyle='--', linewidth=0.7)
#
# plt.xticks(ticks=np.arange(len(unique_param_ids)), labels=param_labels, rotation=90)
# plt.ylabel("Normalized Parameter Value")
# plt.title("Failed Simulations and Base Points by Parameter")
# plt.ylim(0, 1)
# plt.grid(axis='y', linestyle='--', alpha=0.4)
# plt.legend()
# plt.tight_layout()
# plt.show()





# Base setup
step = 183 + 1  # 1 base + 183 perturbations
Result_base = HR[0::step]  # shape: (100,)
X_base = X[0::step, :]  # shape: (100, 183)
bounds = np.array(sp["bounds"])  # shape: (183, 2)

# Normalize all base parameter values
lower_bounds = bounds[:, 0]
upper_bounds = bounds[:, 1]
range_bounds = upper_bounds - lower_bounds

X_base_normalized = (X_base - lower_bounds) / range_bounds  # shape: (100, 183)

# Plot setup
plt.figure(figsize=(12, 5))

for i in range(X_base.shape[0]):  # For each base point (100 total)
# for i in range(100):
    x_vals = np.arange(X_base.shape[1])  # 0 to 182 (parameter indices)
    y_vals = X_base_normalized[i, :]

    color = 'red' if Result_base[i] == 0 else 'green'
    plt.scatter(x_vals, y_vals, color=color, alpha=0.6, s=10,
                label='Failed' if (i == 0 and color == 'red') else 'Successful' if (
                            i == 0 and color == 'green') else "")

# Axis and aesthetics
plt.xlabel("Parameter Index")
plt.ylabel("Normalized Parameter Value")
plt.title("All 100 Base Points Colored by Simulation Success")
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()








































#
# # Normalize: 0 = lower bound, 1 = upper bound
# X_failed_norm = (X_failed - lower_bounds) / (upper_bounds - lower_bounds)
# bad_rows = np.where(X_failed_norm[:, 130] < 0)[0]
# print(X_failed_norm[bad_rows, 130])
#
#
# plt.figure(figsize=(16, 6))
# plt.boxplot(X_failed_norm, vert=True, patch_artist=True, labels=sp["names"], showfliers=False)
# plt.xticks(rotation=90)
# plt.ylabel('Normalized Parameter Value (0 = lower bound, 1 = upper bound)')
# plt.title('Distribution of Parameter Values Causing ODE Failures')
# plt.tight_layout()
# plt.show()



# def find_changed_parameter(X, idx):
#     """Return index of parameter that changed at row idx relative to neighbors."""
#     if idx == 0 or idx == len(X) - 1:
#         return None  # Can't compare both sides
#     before = X[idx - 1]
#     after = X[idx + 1]
#     current = X[idx]
#     changed = np.where((current != before) & (current != after))[0]
#     return changed[0] if len(changed) == 1 else None  # Return only if 1 changed
#
#
# from collections import Counter
#
# fail_indices = np.where(failed_mask)[0]
# fail_param_ids = []
#
# for idx in fail_indices:
#     changed_param = find_changed_parameter(X, idx)
#     if changed_param is not None:
#         fail_param_ids.append(changed_param)
#
# # Count frequency of each failed parameter
# param_failure_count = Counter(fail_param_ids)
#
# # Plot
# param_ids, counts = zip(*sorted(param_failure_count.items()))
# param_labels = [sp["names"][i] for i in param_ids]
#
# plt.figure(figsize=(14, 5))
# plt.bar(param_labels, counts)
# plt.xticks(rotation=90)
# plt.ylabel('Number of Failures')
# plt.title('Number of Simulation Failures by Perturbed Parameter')
# plt.tight_layout()
# plt.show()