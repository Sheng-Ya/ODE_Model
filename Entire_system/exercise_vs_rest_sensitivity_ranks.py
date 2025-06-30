import numpy as np
from SALib import ProblemSpec
from SALib.plotting.bar import plot as barplot
# from SALib.analyze import dgsm
import matplotlib.pyplot as plt
import dgsm_edited as dgsm



X_rest = np.load('DGSM_100_X_samples_HR_P_sys_P_dia_steady_remove.npy')
Result_rest = np.load('DGSM_100_Result_HR_P_sys_P_dia_steady_remove.npy')

X_exercise =  np.load('DGSM_100_X_samples_HR_P_sys_P_dia_exercise_remove.npy', allow_pickle=True)
Result_exercise = np.load('DGSM_100_Result_HR_P_sys_P_dia_exercise_remove.npy', allow_pickle=True)


HR_rest = Result_rest[:, 0]
HR_exercise = Result_exercise[:, 0]

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
        [0.1 * lower, 0 * upper],    [6 * lower, 6 * upper],
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

Si_rest = dgsm.analyze(sp, X_rest, HR_rest, print_to_console=True)
Si_exercise = dgsm.analyze(sp, X_exercise, HR_exercise, print_to_console=True)


## Extract and sort
dgsm_rest = np.array(Si_rest['dgsm'])
names_rest = np.array(Si_rest['names'])
dgsm_exercise = np.array(Si_exercise['dgsm'])
names_exercise = np.array(Si_exercise['names'])



# rank by most influential
dgsm_rest_sorted = dgsm_rest[np.argsort(dgsm_rest)[::-1]]
names_rest_sorted = names_rest[np.argsort(dgsm_rest)[::-1]]
dgsm_exercise_sorted = dgsm_exercise[np.argsort(dgsm_exercise)[::-1]]
names_exercise_sorted = names_exercise[np.argsort(dgsm_exercise)[::-1]]

N = 30 # Change to how many variables to examine

# Get top N names and their Si values at rest
most_important_names = names_rest_sorted[:N]
si_rest_top = dgsm_rest_sorted[:N]

# Get corresponding Si values for exercise (in the same order as rest's top N)
# Make a dictionary for fast lookup
exercise_dict = dict(zip(names_exercise, dgsm_exercise))
si_exercise_top = np.array([exercise_dict[name] for name in most_important_names])

# X-axis labels
labels = most_important_names
x = np.arange(len(labels))
width = 0.35

# Plot
fig, ax = plt.subplots(figsize=(8, 12))
ax.barh(x - width/2, np.log(si_rest_top), width, label='Rest')
ax.barh(x + width/2, np.log(si_exercise_top), width, label='Exercise')

ax.set_xlabel('log(Sensitivity Index (Si))')
ax.set_title('Top DGSM Sensitivities for HR at Rest and Exercise')
ax.set_yticks(x)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # Most influential at top
ax.legend()
plt.tight_layout()
plt.show()





# Define range for next 50 most influential parameters
start_idx = 133
end_idx = 183

# Get next 50 parameter names and Si values at rest
next_important_names = names_rest_sorted[start_idx:end_idx]
si_rest_next = dgsm_rest_sorted[start_idx:end_idx]

# Get corresponding Si values for exercise
si_exercise_next = np.array([exercise_dict[name] for name in next_important_names])

# X-axis labels
labels_next = next_important_names
x_next = np.arange(len(labels_next))

# Plot
fig, ax = plt.subplots(figsize=(8, 14))
ax.barh(x_next - width/2, si_rest_next, width, label='Rest')
ax.barh(x_next + width/2, si_exercise_next, width, label='Exercise')

ax.set_xlabel('Sensitivity Index')
ax.set_title('DGSM Sensitivities for HR (last 50 parameters)')
ax.set_yticks(x_next)
ax.set_yticklabels(labels_next)
ax.invert_yaxis()  # So more important parameters are higher up
ax.legend()
plt.tight_layout()
plt.show()












def get_ranks(names_reference, target_names):
    return [np.where(names_reference == name)[0][0] + 1 for name in target_names]  # +1 for 1-based rank

N = 50 # Change to how many variables to examine

# plot most influential
most_important_names = names_rest_sorted[:N]

# Create a name → rank mapping for DGSM_250
dgsm_rest_ranks = {name: rank + 1 for rank, name in enumerate(names_rest_sorted)}

# Sort least_important_names by their DGSM_250 rank (ascending = least to most influential)
most_important_names_sorted = most_important_names

# Update all relevant lists in the same order
ranks_exercise = get_ranks(names_exercise_sorted, most_important_names_sorted)
ranks_rest = get_ranks(names_rest_sorted, most_important_names_sorted)

# X-axis labels with DGSM_250 rank
labels = [f"{name} ({dgsm_rest_ranks[name]})" for name in most_important_names_sorted]


# X-axis labels
labels = most_important_names
x = np.arange(len(labels))
width = 0.35

# Plot
fig, ax = plt.subplots(figsize=(8, 12))
ax.barh(x - width/2, ranks_rest, width, label='Rest', color='lightblue')
ax.barh(x + width/2, ranks_exercise, width, label='Exercise', color='#ff7f0e')

ax.set_xlabel('Ranks')
ax.set_title('Top DGSM Sensitivities for HR at Rest and Exercise')
ax.set_yticks(x)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # Most influential at top
ax.legend()
plt.tight_layout()
plt.show()










# Number of parameters to plot
N = 50

# Select the least influential parameters (last N from sorted list)
least_important_names = names_rest_sorted[-N:]

# Create a name → rank mapping from the rest condition
dgsm_rest_ranks = {name: rank + 1 for rank, name in enumerate(names_rest_sorted)}

# Get ranks in both conditions
ranks_rest_least = get_ranks(names_rest_sorted, least_important_names)
ranks_exercise_least = get_ranks(names_exercise_sorted, least_important_names)

# Labels with ranks for better readability
labels_least = [f"{name} ({dgsm_rest_ranks[name]})" for name in least_important_names]

# X-axis setup
x = np.arange(len(labels_least))
width = 0.35

# Plot
fig, ax = plt.subplots(figsize=(8, 14))
ax.barh(x - width/2, ranks_rest_least, width, label='Rest', color='lightblue')
ax.barh(x + width/2, ranks_exercise_least, width, label='Exercise', color='#ff7f0e')

ax.set_xlabel('Rank (lower = more influential)')
ax.set_title('Ranks of Least Influential DGSM Parameters for HR')
ax.set_yticks(x)
ax.set_yticklabels(labels_least)
ax.invert_yaxis()  # So lower ranks appear higher
ax.legend()
plt.tight_layout()
plt.show()













#
# # Linear fitting variables interpreted
# # Organize into a dict: param → {"LOW": (..), "HIGH": (..)}
# param_data = {}
# for label, values in Result_linear:
#     param_name, level = label.split()
#     if param_name not in param_data:
#         param_data[param_name] = {}
#     param_data[param_name][level] = values
#
# diffs_hr = {}
#
# for param, levels in param_data.items():
#     if "LOW" in levels and "HIGH" in levels:
#         low_vals = np.array(levels["LOW"])
#         high_vals = np.array(levels["HIGH"])
#         diff = np.abs(high_vals - low_vals)
#
#         diffs_hr[param] = diff[0]
#     else:
#         print(f"Warning: missing HIGH/LOW for {param}")
#
# # Sort by descending difference
# sorted_params_hr = sorted(diffs_hr, key=diffs_hr.get, reverse=True)
# sorted_values_hr = [diffs_hr[k] for k in sorted_params_hr]
#
# # Create linear rank mapping
# ranks_linear_dict = {name: rank + 1 for rank, name in enumerate(sorted_params_hr)}







# N = 80  # Change to how many least significant you want to examine
# least_important_names = names_250_sorted[-N:]
#
# def get_ranks(names_reference, target_names):
#     return [np.where(names_reference == name)[0][0] + 1 for name in target_names]  # +1 for 1-based rank
#
#
#
# # Create a name → rank mapping for DGSM_250
# dgsm_250_ranks = {name: rank + 1 for rank, name in enumerate(names_250_sorted)}
#
# # Sort least_important_names by their DGSM_250 rank (ascending = least to most influential)
# least_important_names_sorted = least_important_names
#
# # Update all relevant lists in the same order
# ranks_4 = get_ranks(names_4_sorted, least_important_names_sorted)
# ranks_10 = get_ranks(names_10_sorted, least_important_names_sorted)
# ranks_250 = get_ranks(names_250_sorted, least_important_names_sorted)
# ranks_linear = [ranks_linear_dict[name] for name in least_important_names_sorted]
#
# # X-axis labels with DGSM_250 rank
# labels = [f"{name} ({dgsm_250_ranks[name]})" for name in least_important_names_sorted]
#
#
# x = np.arange(len(labels))  # label locations
# width = 0.2
#
# fig, ax = plt.subplots(figsize=(6, 15))
# rects1 = ax.barh(x - 1.5*width, ranks_linear, width, label='Linear ΔHR')
# rects2 = ax.barh(x - 0.5*width, ranks_4, width, label='DGSM_4')
# rects3 = ax.barh(x + 0.5*width, ranks_10, width, label='DGSM_10')
#
# ax.set_xlabel('Rank (lower = more important)')
# ax.set_title('Ranks of Least Sensitive DGSM_250 Parameters for HR at REST')
# ax.set_yticks(x)
# ax.set_yticklabels(labels)
# ax.legend()
# ax.set_ylim(-0.5, len(labels) - 0.5)
# plt.tight_layout()
# # plt.gca().invert_yaxis()  # So that lower ranks are higher up
# plt.show()