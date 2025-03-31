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

X = np.load('X_samples_900.npy')
Result = np.load('Result_900.npy')

# Generate samples
# param_values = sample(problem, 1024)

# compare emulators
ae = AutoEmulate()
ae.setup(X, Result)

best_emulator = ae.compare()

ae.summarise_cv()

ae.save(best_emulator, "best_emulator")
final_loaded = ae.load("best_emulator")

ae.plot_cv()
ae.plot_cv(style="actual_vs_predicted")
ae.plot_cv(style="residual_vs_predicted")

