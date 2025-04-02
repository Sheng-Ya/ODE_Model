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

X = np.load('X_samples_P_sys_1000.npy')
Result = np.load('Result_P_sys_1000.npy')

# Generate samples
# param_values = sample(problem, 1024)

# compare emulators
ae = AutoEmulate()
ae.setup(X, Result)

best_emulator = ae.compare()

ae.summarise_cv()

ae.plot_cv()
ae.plot_cv(style="actual_vs_predicted")
ae.plot_cv(style="residual_vs_predicted")


#
# test set results for the best emulator
ae.evaluate(best_emulator)
ae.plot_eval(best_emulator)

# Run model (example)
emulator = ae.refit(best_emulator)

ae.save(emulator, "best_emulator_p_sys")

#
# (sp.sample_saltelli(1024, calc_second_order = True)
# .evaluate(emulator.predict)
# .analyze_sobol()
#  )
#
# # Y = emulator.predict(param_values)
#
# # Perform analysis
# # Si = sobol.analyze(problem, Y, print_to_console=True)
#
# sp.plot()
#
# plt.title("Basic example plot")
#
# axes = sp.plot()
# for ax in axes:
#     ax.set_yscale("log")
#
# axes[0].set_title("Example custom plot with log scale")
#
# # Other custom layouts can be created in the usual matplotlib style
# # with the basic bar plotter.
#
# # Example: Direct control of plot elements
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 16))
#
# # Get result DataFrames
# total, first = sp.to_df()
#
# ax1 = barplot(total, ax=ax1)
# ax2 = barplot(first, ax=ax2)
# # ax3 = barplot(second, ax=ax3)
#
# ax1.set_yscale("log")
# ax2.set_yscale("log")
#
# ax1.set_title("Customized matplotlib plot")
# plt.show()
#
#
# # Plot sensitivity indices as a heatmap
# # Note that plotting methods return a matplotlib axes object
# ax = sp.heatmap("CO")
# ax.set_title("Basic heatmap")
# plt.show()
#
#
# # Another heatmap plot with more fine-grain control
# # Displays Total and First-Order sensitivities in separate subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
# sp.heatmap("CO", "ST", "Total Order Sensitivity", ax1)
# sp.heatmap("CO", "S1", "First Order Sensitivity", ax2)
# plt.show()
#
#
# # Yet another heatmap example
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
#     2, 2, figsize=(10, 6), sharex=True, constrained_layout=True
# )
# sp.heatmap("CO", "ST", "Total Order", ax=ax1)
# sp.heatmap("CO", "ST_conf", "Total Order Conf.", ax=ax2)
# sp.heatmap("CO", "S1", "First Order", ax=ax3)
# sp.heatmap("CO", "S1_conf", "First Order Conf.", ax=ax4)
# plt.show()
#
