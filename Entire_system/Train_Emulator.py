import os
import joblib
import torch
from SALib import ProblemSpec
from SALib.plotting.bar import plot as barplot
from SALib.analyze import sobol
from SALib.analyze.sobol import analyze
# from SALib.sample import saltelli
from SALib.sample.sobol import sample
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from autoemulate import AutoEmulate

from sklearn.model_selection import KFold

X = np.load('All_params_LHCS_200000_X_sample_HR_Plv_Prv_Vlv_Vrv_rest.npy') # size [N, 299]
Result_all = np.load('All_params_LHCS_200000_Result_HR_Plv_Prv_Vlv_Vrv_rest.npy') # size [N, 9]

# Stroke_Volume = Result_all[:, 3] - Result_all[:, 4]
# Ejection_fraction = (Stroke_Volume / Result_all[:, 3]) * 100

# Result_all = np.column_stack((Result_all, Stroke_Volume))
# Result_all = np.column_stack((Result_all, Ejection_fraction))

mask = Result_all[:,0] != 0

X = X[mask, :]
Result = Result_all[mask, :]

# choose which results (column) to look at
Result_cols = ["Heart Rate", "Systolic Pressure", "Diastolic Pressure", "EDV", "ESV", "Max RV Volume", "Min RV Volume",
               "Max RV Pressure", "Min RV Pressure", "Stroke Volume", "Ejection Fraction"]
Result = Result[:, 0] # Heart rate here

## EMULATION
idx = np.random.choice(len(Result), size=60000, replace=False)
X = X[idx,:]
Result = Result[idx]


# compare emulators
ae = AutoEmulate(X, Result, log_level="error", models=["rbf"], device="cuda")
# ae.setup(X, Result,  param_search=True, param_search_iters=30, n_jobs=-1, models=["rbf"], cross_validator=KFold(n_splits=3))
# ae.setup(X, Result, n_jobs=-1, models=["rbf"])
ae.summarise()
best = ae.best_result()
print(best.params)

fig = ae.plot(best, fname="best_model_plot.png")


# fig = ae.plot_cv()
# fig = ae.plot_cv(style="residual_vs_predicted")
# fig.savefig("cv_plot.png", dpi=300, bbox_inches="tight")  # save figure

# rbf = ae.best_result()
# fig.savefig("eval_plot.png", dpi=300, bbox_inches="tight")  # save figure
#
# rbf_final = ae.refit(rbf)

# ae.save(best, "rbf_new")
os.makedirs("rbf1", exist_ok=True)
joblib.dump(best, "rbf1/RBF.joblib")

# gp = ae.get_model("GaussianProcess")
# ae.evaluate(gp)
# ae.plot_eval(gp)
# gp_final = ae.refit(gp)

# os.makedirs("gp_final", exist_ok=True)
# joblib.dump(gp_final, "gp_final/GaussianProcess.joblib")

# ae.save(gp_final, "gp_final")