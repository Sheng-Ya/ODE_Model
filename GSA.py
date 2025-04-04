from SALib import ProblemSpec
from SALib.plotting.bar import plot as barplot
from SALib.analyze import sobol
from SALib.analyze.sobol import analyze
# from SALib.sample import saltelli
from SALib.sample.sobol import sample
from SALib.test_functions import Ishigami
import matplotlib.pyplot as plt
import numpy as np
from autoemulate.compare import AutoEmulate
from autoemulate.logging_config import _configure_logging

lower = 0.5
upper = 1.5

# Define the model inputs
# sp = ProblemSpec({
#     'outputs': ["P_sys"],
#     # 'num_vars': 90,  # Number of parameters
#     'names': [
#         'C_sa', 'L_sa', 'R_sa', 'Vu_sa', 'C_amp', 'C_amv', 'C_bp', 'C_bv', 'C_ep', 'C_ev',
#         'C_hp', 'C_hv', 'C_rmp', 'C_rmv', 'C_sp', 'C_sv', 'kr_am', 'P_0', 'R_amv_n', 'R_bv_n',
#         'R_ev_n', 'R_hv_n', 'R_rmv_n', 'R_sv_n', 'V_tot', 'Vu_amp', 'Vu_bp', 'Vu_bv', 'Vu_ep',
#         'Vu_hp', 'Vu_hv', 'Vu_rmp', 'Vu_sp', 'D1', 'D2', 'K1_vc', 'K2_vc', 'Kr_vc', 'Rvc_n',
#         'Vu_vc', 'Vvc_max', 'Vvc_min', 'C_pa', 'C_pp', 'C_pv', 'L_pa', 'R_pa', 'R_pp', 'R_pv',
#         'Vu_pa', 'Vu_pp', 'Vu_pv', 'KE_lv', 'KE_rv', 'Emax_la', 'P0_la', 'KE_la', 'Emax_ra',
#         'P0_ra', 'KE_ra', 'P0_lv', 'P0_rv', 'Vu_la', 'Vu_lv', 'Vu_ra', 'Vu_rv', 'g_abd', 'g_thor',
#         'P_abdmax_n', 'P_abdmin_n', 'P_thormax_n', 'P_thormin_n', 'VT_n', 'BF', 'Emax_lv',
#         'Emax_rv', 'HR', 'T_resp', 'TI', 'VT', 'Vu_ev', 'Vu_amv', 'Vu_rmv', 'Vu_sv', 'R_ep',
#         'R_amp', 'R_rmp', 'R_sp', 'R_bp', 'R_hp'
#     ],
#     'bounds': [
#         [0.28 * lower, 0.28 * upper], [0.00022 * lower, 0.00022 * upper], [0.06 * lower, 0.06 * upper],
#         [0.0 * lower, 10.0 * upper], [0.315 * lower, 0.315 * upper], [9.4 * lower, 9.4 * upper],
#         [0.358 * lower, 0.358 * upper], [10.71 * lower, 10.71 * upper], [0.668 * lower, 0.668 * upper],
#         [20 * lower, 20 * upper], [0.119 * lower, 0.119 * upper], [3.57 * lower, 3.57 * upper],
#         [0.21 * lower, 0.21 * upper], [6.28 * lower, 6.28 * upper], [2.05 * lower, 2.05 * upper],
#         [61.11 * lower, 61.11 * upper], [24.17 * lower, 24.17 * upper], [3.93 * lower, 3.93 * upper],
#         [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper],
#         [0.224 * lower, 0.224 * upper], [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper],
#         [5027.6 * lower, 5027.6 * upper], [60.22 * lower, 60.22 * upper], [68.42 * lower, 68.42 * upper],
#         [279.49 * lower, 279.49 * upper], [127.72 * lower, 127.72 * upper], [23 * lower, 23 * upper],
#         [93.16 * lower, 93.16 * upper], [40.1 * lower, 40.1 * upper], [260.3 * lower, 260.3 * upper],
#         [0.3855 * lower, 0.3855 * upper], [-5 * upper, -5 * lower], [0.15 * lower, 0.15 * upper],
#         [0.4 * lower, 0.4 * upper], [0.001 * lower, 0.001 * upper], [0.025 * lower, 0.025 * upper],
#         [123 * lower, 123 * upper], [350 * lower, 350 * upper], [50 * lower, 50 * upper],
#         [0.76 * lower, 0.76 * upper], [5.8 * lower, 5.8 * upper], [25.37 * lower, 25.37 * upper],
#         [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
#         [0.0056 * lower, 0.0056 * upper], [0.0 * lower, 10.0 * upper], [116.6775 * lower, 116.6775 * upper],
#         [114 * lower, 114 * upper], [0.014 * lower, 0.014 * upper], [0.011 * lower, 0.011 * upper],
#         [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper], [0.05 * lower, 0.05 * upper],
#         [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper], [0.05 * lower, 0.05 * upper],
#         [1.5 * lower, 1.5 * upper], [1.5 * lower, 1.5 * upper], [24 * lower, 24 * upper],
#         [15.908 * lower, 15.908 * upper], [24 * lower, 24 * upper], [38.703 * lower, 38.703 * upper],
#         [3.39 * lower, 3.39 * upper], [6.8 * lower, 6.8 * upper], [-1 * upper, 0 * lower],
#         [-2.5 * upper, -2.5 * lower], [-1 * upper, 0.0 * lower], [-3 * upper, 0.0 * lower],
#         [0.73 * lower, 0.73 * upper], [0.25 * lower, 0.25 * upper], [2.392 * lower, 2.392 * upper],
#         [1.412 * lower, 1.412 * upper], [1.2 * lower, 1.2 * upper], [4 * lower, 4 * upper],
#         [1.8 * lower, 1.8 * upper], [0.73 * lower, 0.73 * upper], [607.8 * lower, 607.8 * upper],
#         [286.4 * lower, 286.4 * upper], [190.95 * lower, 190.95 * upper], [1361.6 * lower, 1361.6 * upper],
#         [1.655 * lower, 1.655 * upper], [3.51 * lower, 3.51 * upper], [5.27 * lower, 5.27 * upper],
#         [2.49 * lower, 2.49 * upper], [6.57 * lower, 6.57 * upper], [19.71 * lower, 19.71 * upper]
#     ],
# })


sp = ProblemSpec({
    'outputs': ["P_sys"],
    # 'num_vars': 90,  # Number of parameters
    'names': [
        'C_sa', 'L_sa', 'R_sa', 'C_amp', 'C_amv', 'C_bp', 'C_bv', 'C_ep', 'C_ev',
        'C_hp', 'C_hv', 'C_rmp', 'C_rmv', 'C_sp', 'C_sv', 'R_amv_n', 'R_bv_n',
        'R_ev_n', 'R_hv_n', 'R_rmv_n', 'R_sv_n', 'D1', 'D2', 'K1_vc', 'K2_vc',
        'Kr_vc', 'Rvc_n', 'C_pa', 'C_pp', 'C_pv', 'L_pa', 'R_pa', 'R_pp', 'R_pv',
        'Emax_la', 'P0_la', 'Emax_ra', 'P0_ra', 'P0_lv', 'P0_rv', 'g_abd', 'g_thor',
        'P_abdmax_n', 'P_abdmin_n', 'P_thormax_n', 'P_thormin_n', 'VT_n', 'BF',
        'Emax_lv', 'Emax_rv', 'HR', 'T_resp', 'TI', 'VT', 'R_ep', 'R_amp', 'R_rmp',
        'R_sp', 'R_bp', 'R_hp'
    ],
    'bounds': [
        [0.28 * lower, 0.28 * upper], [0.00022 * lower, 0.00022 * upper], [0.06 * lower, 0.06 * upper],
        [0.315 * lower, 0.315 * upper], [9.4 * lower, 9.4 * upper], [0.358 * lower, 0.358 * upper],
        [10.71 * lower, 10.71 * upper], [0.668 * lower, 0.668 * upper], [20 * lower, 20 * upper],
        [0.119 * lower, 0.119 * upper], [3.57 * lower, 3.57 * upper], [0.21 * lower, 0.21 * upper],
        [6.28 * lower, 6.28 * upper], [2.05 * lower, 2.05 * upper], [61.11 * lower, 61.11 * upper],
        [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper],
        [0.224 * lower, 0.224 * upper], [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper],
        [0.3855 * lower, 0.3855 * upper], [-5 * upper, -5 * lower], [0.15 * lower, 0.15 * upper],
        [0.4 * lower, 0.4 * upper], [0.001 * lower, 0.001 * upper], [0.025 * lower, 0.025 * upper],
        [0.76 * lower, 0.76 * upper], [5.8 * lower, 5.8 * upper], [25.37 * lower, 25.37 * upper],
        [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
        [0.0056 * lower, 0.0056 * upper], [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper],
        [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper], [1.5 * lower, 1.5 * upper],
        [1.5 * lower, 1.5 * upper], [3.39 * lower, 3.39 * upper], [6.8 * lower, 6.8 * upper],
        [-1 * upper, 0 * lower], [-2.5 * upper, -2.5 * lower], [-1 * upper, 0.0 * lower],
        [-3 * upper, 0.0 * lower], [0.73 * lower, 0.73 * upper], [0.25 * lower, 0.25 * upper],
        [2.392 * lower, 2.392 * upper], [1.412 * lower, 1.412 * upper], [1.2 * lower, 1.2 * upper],
        [4 * lower, 4 * upper], [1.8 * lower, 1.8 * upper], [0.73 * lower, 0.73 * upper],
        [1.655 * lower, 1.655 * upper], [3.51 * lower, 3.51 * upper], [5.27 * lower, 5.27 * upper],
        [2.49 * lower, 2.49 * upper], [6.57 * lower, 6.57 * upper], [19.71 * lower, 19.71 * upper]
    ],
})





ae = AutoEmulate()
ae.logger = _configure_logging()
best_emulator = ae.load("best_emulator_p_sys_fixed")


#
# # test set results for the best emulator
# ae.evaluate(best_emulator)
# ae.plot_eval(best_emulator)
#
# # Run model (example)
# emulator = ae.refit(best_emulator)


(sp.sample_saltelli(1024, calc_second_order = True)
.evaluate(best_emulator.predict)
.analyze_sobol()
 )

# Y = emulator.predict(param_values)

# Perform analysis
# Si = sobol.analyze(problem, Y, print_to_console=True)

sp.plot()

plt.title("Basic example plot")

axes = sp.plot()
for ax in axes:
    ax.set_yscale("log")

axes[0].set_title("Example custom plot with log scale")
plt.show()

# Other custom layouts can be created in the usual matplotlib style
# with the basic bar plotter.

# Example: Direct control of plot elements
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 16))

# Get result DataFrames
total, first, second = sp.to_df()

ax1 = barplot(total, ax=ax1)
ax2 = barplot(first, ax=ax2)
# ax3 = barplot(second, ax=ax3)

ax1.set_yscale("log")
ax2.set_yscale("log")

ax1.set_title("Customized matplotlib plot")
plt.show()


# Plot sensitivity indices as a heatmap
# Note that plotting methods return a matplotlib axes object
ax = sp.heatmap("P_sys")
ax.set_title("Basic heatmap")
plt.show()


# Another heatmap plot with more fine-grain control
# Displays Total and First-Order sensitivities in separate subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
sp.heatmap("P_sys", "ST", "Total Order Sensitivity", ax1)
sp.heatmap("P_sys", "S1", "First Order Sensitivity", ax2)
plt.show()


# Yet another heatmap example
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, figsize=(10, 6), sharex=True, constrained_layout=True
)
sp.heatmap("P_sys", "ST", "Total Order", ax=ax1)
sp.heatmap("P_sys", "ST_conf", "Total Order Conf.", ax=ax2)
sp.heatmap("P_sys", "S1", "First Order", ax=ax3)
sp.heatmap("P_sys", "S1_conf", "First Order Conf.", ax=ax4)
plt.show()