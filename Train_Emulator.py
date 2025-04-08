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

X = np.load('C:/Users/vanes/Downloads/exercise_model/ODE_Exercise/Entire system/X_samples_PA_gas_resp.npy')
two_Result = np.load('C:/Users/vanes/Downloads/exercise_model/ODE_Exercise/Entire system/Result_PA_gas_resp.npy')

Result = two_Result[:, 0]
#
# X1 = np.load('X_samples_PaO2_1000.npy')
# Result1 = np.load('Result_PaO2_1000.npy')

# Generate samples
# param_values = sample(problem, 1024)

# compare emulators
ae = AutoEmulate()
ae.setup(X, Result)

# sp = AutoEmulate()
# sp.setup(X1, Result1)

best_emulator = ae.compare()
# best_emulator1 = sp.compare()

ae.summarise_cv()
# sp.summarise_cv()

ae.plot_cv()
ae.plot_cv(style="actual_vs_predicted")
ae.plot_cv(style="residual_vs_predicted")

# RB = ae.get_model("RadialBasisFunctions")
gp = ae.get_model("GaussianProcess")
# ae.evaluate(RB)
ae.evaluate(gp)
#
# test set results for the best emulator
ae.plot_eval(gp)

# sp.evaluate(best_emulator1)

# Run model (example)
gp_final = ae.refit(gp)
# rb_final = ae.refit(RB)
# emulator1 = sp.refit(best_emulator1)

# ae.save(rb_final, "best_emulator_PaO2_fixed")
# ae.save(gp_final, "best_emulator_PaO2_fixed1")




lower = 0.5
upper = 1.5


# sp = ProblemSpec({
#     'outputs': ["PaO2"],
#     # 'num_vars': 90,  # Number of parameters
#     'names': [
#         'C_sa', 'L_sa', 'R_sa', 'C_amp', 'C_amv', 'C_bp', 'C_bv', 'C_ep', 'C_ev',
#         'C_hp', 'C_hv', 'C_rmp', 'C_rmv', 'C_sp', 'C_sv', 'R_amv_n', 'R_bv_n',
#         'R_ev_n', 'R_hv_n', 'R_rmv_n', 'R_sv_n', 'D1', 'D2', 'K1_vc', 'K2_vc',
#         'Kr_vc', 'Rvc_n', 'C_pa', 'C_pp', 'C_pv', 'L_pa', 'R_pa', 'R_pp', 'R_pv',
#         'Emax_la', 'P0_la', 'Emax_ra', 'P0_ra', 'P0_lv', 'P0_rv', 'g_abd', 'g_thor',
#         'P_abdmax_n', 'P_abdmin_n', 'P_thormax_n', 'P_thormin_n', 'VT_n', 'BF',
#         'Emax_lv', 'Emax_rv', 'HR', 'T_resp', 'TI', 'VT', 'R_ep', 'R_amp', 'R_rmp',
#         'R_sp', 'R_bp', 'R_hp',
#
#         'LCTV', 'T1', 'T2', 'VL_CO2', 'VL_O2', 'Z',
#         'dc', 'h', 'KCCO2', 'KCSFCO2', 'SbCO2', 'SCO2',
#         'VTCO2', 'VTO2',
#         'E_CW', 'E_L', 'k_aw1', 'k_aw2', 'P_ao', 'R_rs',
#         'A0_ua', 'C_ua', 'K_ua', 'Pcrit_min', 'R_CW', 'R_trachea',
#         'GV_dead', 'Kbg', 'KcCO2', 'KcMRV', 'KpCO2', 'KpO2', 'V0_dead', 'VA_rest'
#     ],
#     'bounds': [
#         [0.28 * lower, 0.28 * upper], [0.00022 * lower, 0.00022 * upper], [0.06 * lower, 0.06 * upper],
#         [0.315 * lower, 0.315 * upper], [9.4 * lower, 9.4 * upper], [0.358 * lower, 0.358 * upper],
#         [10.71 * lower, 10.71 * upper], [0.668 * lower, 0.668 * upper], [20 * lower, 20 * upper],
#         [0.119 * lower, 0.119 * upper], [3.57 * lower, 3.57 * upper], [0.21 * lower, 0.21 * upper],
#         [6.28 * lower, 6.28 * upper], [2.05 * lower, 2.05 * upper], [61.11 * lower, 61.11 * upper],
#         [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper],
#         [0.224 * lower, 0.224 * upper], [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper],
#         [0.3855 * lower, 0.3855 * upper], [-5 * upper, -5 * lower], [0.15 * lower, 0.15 * upper],
#         [0.4 * lower, 0.4 * upper], [0.001 * lower, 0.001 * upper], [0.025 * lower, 0.025 * upper],
#         [0.76 * lower, 0.76 * upper], [5.8 * lower, 5.8 * upper], [25.37 * lower, 25.37 * upper],
#         [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
#         [0.0056 * lower, 0.0056 * upper], [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper],
#         [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper], [1.5 * lower, 1.5 * upper],
#         [1.5 * lower, 1.5 * upper], [3.39 * lower, 3.39 * upper], [6.8 * lower, 6.8 * upper],
#         [-1 * upper, 0 * lower], [-2.5 * upper, -2.5 * lower], [-1 * upper, 0.0 * lower],
#         [-3 * upper, 0.0 * lower], [0.73 * lower, 0.73 * upper], [0.25 * lower, 0.25 * upper],
#         [2.392 * lower, 2.392 * upper], [1.412 * lower, 1.412 * upper], [1.2 * lower, 1.2 * upper],
#         [4 * lower, 4 * upper], [1.8 * lower, 1.8 * upper], [0.73 * lower, 0.73 * upper],
#         [1.655 * lower, 1.655 * upper], [3.51 * lower, 3.51 * upper], [5.27 * lower, 5.27 * upper],
#         [2.49 * lower, 2.49 * upper], [6.57 * lower, 6.57 * upper], [19.71 * lower, 19.71 * upper],
#
#         [0.588 * lower, 0.588 * upper],
#         [1 * lower, 1 * upper],
#         [2 * lower, 2 * upper],
#         [3 * lower, 3 * upper],
#         [2.5 * lower, 2.5 * upper],
#         [0.0227 * lower, 0.0227 * upper],
#
#         [0.015 * lower, 0.015 * upper],
#         [(0.0183 / 1000) * lower, (0.0183 / 1000) * upper],
#         [346000 * lower, 346000 * upper],
#         [320 * lower, 320 * upper],
#         [(0.36 / 1000) * lower, (0.36 / 1000) * upper],
#         [0.0043 * lower, 0.0043 * upper],
#
#         [0.25 * lower, 0.25 * upper],
#         [0.25 * lower, 0.25 * upper],
#
#         [10.545 * lower, 10.545 * upper],
#         [10.545 * lower, 10.545 * upper],
#         [1.85 * lower, 1.85 * upper],
#         [0.43 * lower, 0.43 * upper],
#         [0.0 * lower, 1.0 * upper],
#         [3.02 * 0.73559 * lower, 3.02 * 0.73559 * upper],
#
#         [1 * lower, 1 * upper],
#         [(0.001 / 0.73559) * lower, (0.001 / 0.73559) * upper],
#         [(1 / 0.73559) * lower, (1 / 0.73559) * upper],
#         [(-40 * 0.73559) * upper, (-40 * 0.73559) * lower],
#         [0.8326 * 0.73559 * lower, 0.8326 * 0.73559 * upper],
#         [1000000 * 0.73559 * lower, 1000000 * 0.73559 * upper],
#
#         [0.1698 * lower, 0.1698 * upper],
#         [17.4 * lower, 17.4 * upper],
#         [0.2332 * lower, 0.2332 * upper],
#         [1 * lower, 1 * upper],
#         [0.2025 * lower, 0.2025 * upper],
#         [4.72e-9 * lower, 4.72e-9 * upper],
#         [0.1587 * lower, 0.1587 * upper],
#         [0.0673 * lower, 0.0673 * upper]
#     ],
# })



sp = ProblemSpec({
    'outputs': ["PaO2"],
    # 'num_vars': 90,  # Number of parameters
    'names': [
        "LCTV", "T1", "T2", "VL_CO2", "VL_O2",
        "dc", "h", "KCCO2", "KCSFCO2", "SbCO2", "SCO2",
        "VTCO2", "VTO2",
        "E_CW", "E_L", "k_aw1", "k_aw2", "P_ao", "R_rs",
        "A0_ua", "C_ua", "K_ua", "Pcrit_min", "R_CW", "R_trachea",
        "GV_dead", "Kbg", "KcCO2", "KcMRV", "KpCO2", "KpO2"
    ],
    'bounds': [
            [0.588 * lower, 0.588 * upper],  # LCTV
            [1 * lower, 1 * upper],  # T1
            [2 * lower, 2 * upper],  # T2
            [3 * lower, 3 * upper],  # VL_CO2
            [2.5 * lower, 2.5 * upper],  # VL_O2
            # [0.0227 * lower, 0.0227 * upper],       # Z — commented out

            [0.015 * lower, 0.015 * upper],  # dc
            [(0.0183 / 1000) * lower, (0.0183 / 1000) * upper],  # h
            [346000 * lower, 346000 * upper],  # KCCO2
            [320 * lower, 320 * upper],  # KCSFCO2
            [(0.36 / 1000) * lower, (0.36 / 1000) * upper],  # SbCO2
            [0.0043 * lower, 0.0043 * upper],  # SCO2

            [0.25 * lower, 0.25 * upper],  # VTCO2
            [0.25 * lower, 0.25 * upper],  # VTO2

            [10.545 * lower, 10.545 * upper],  # E_CW
            [10.545 * lower, 10.545 * upper],  # E_L
            [1.85 * lower, 1.85 * upper],  # k_aw1
            [0.43 * lower, 0.43 * upper],  # k_aw2
            [0.0 * lower, 1.0 * upper],  # P_ao
            [3.02 * lower, 3.02 * upper],  # R_rs

            [1 * lower, 1 * upper],  # A0_ua
            [0.001 * lower, 0.001 * upper],  # C_ua (converted)
            [1 * lower, 1 * upper],  # K_ua (converted)
            [-40 * upper, -40 * lower],  # Pcrit_min (converted)
            [0.8326 * lower, 0.8326 * upper],  # R_CW (converted)
            [1000000 * lower, 1000000 * upper],  # R_trachea (converted)

            [0.1698 * lower, 0.1698 * upper],  # GV_dead
            [17.4 * lower, 17.4 * upper],  # Kbg
            [0.2332 * lower, 0.2332 * upper],  # KcCO2
            [1 * lower, 1 * upper],  # KcMRV
            [0.2025 * lower, 0.2025 * upper],  # KpCO2
            [4.72e-9 * lower, 4.72e-9 * upper],  # KpO2
        ],
})




(sp.sample_saltelli(1024, calc_second_order = True)
.evaluate(gp_final.predict)
.analyze_sobol()
 )

# Y = emulator.predict(param_values)

# Perform analysis
# Si = sobol.analyze(problem, Y, print_to_console=True)

# sp.plot()

# plt.title("PaO2")

# axes = sp.plot()
# for ax in axes:
#     ax.set_yscale("log")
#
# axes[0].set_title("Example custom plot with log scale")
# plt.show()

# Other custom layouts can be created in the usual matplotlib style
# with the basic bar plotter.

# Example: Direct control of plot elements
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 16))

# Get result DataFrames
total, first, second = sp.to_df()

ax1 = barplot(total, ax=ax1)
ax2 = barplot(first, ax=ax2)
# ax3 = barplot(second, ax=ax3)

ax1.set_title("PaO2 plot")
plt.show()


# Plot sensitivity indices as a heatmap
# Note that plotting methods return a matplotlib axes object
ax = sp.heatmap("PaO2")
ax.set_title("Basic heatmap")
plt.show()


# Another heatmap plot with more fine-grain control
# Displays Total and First-Order sensitivities in separate subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
sp.heatmap("PaO2", "ST", "Total Order Sensitivity", ax1)
sp.heatmap("PaO2", "S1", "First Order Sensitivity", ax2)
plt.show()


# Yet another heatmap example
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, figsize=(10, 6), sharex=True, constrained_layout=True
)
sp.heatmap("PaO2", "ST", "Total Order", ax=ax1)
sp.heatmap("PaO2", "ST_conf", "Total Order Conf.", ax=ax2)
sp.heatmap("PaO2", "S1", "First Order", ax=ax3)
sp.heatmap("PaO2", "S1_conf", "First Order Conf.", ax=ax4)
plt.show()