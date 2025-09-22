import os
from SALib import ProblemSpec
from SALib.plotting.bar import plot as barplot
from SALib.analyze import sobol
from SALib.analyze.sobol import analyze
# from SALib.sample import saltelli
from SALib.sample.sobol import sample
import matplotlib.pyplot as plt
import numpy as np
from autoemulate.compare import AutoEmulate
from sklearn.model_selection import KFold

X = np.load('All_params_LHCS_200000_X_sample_HR_Plv_Prv_Vlv_Vrv_rest.npy') # size [N, 299]
Result_all = np.load('All_params_LHCS_200000_Result_HR_Plv_Prv_Vlv_Vrv_rest.npy') # size [N, 9]

Stroke_Volume = Result_all[:, 3] - Result_all[:, 4]
Ejection_fraction = (Stroke_Volume / Result_all[:, 3]) * 100

Result_all = np.column_stack((Result_all, Stroke_Volume))
Result_all = np.column_stack((Result_all, Ejection_fraction))

mask = Result_all[:,0] != 0

X = X[mask, :]
Result = Result_all[mask, :]

# choose which results (column) to look at
Result_cols = ["Heart Rate", "Systolic Pressure", "Diastolic Pressure", "EDV", "ESV", "Max RV Volume", "Min RV Volume",
               "Max RV Pressure", "Min RV Pressure", "Stroke Volume", "Ejection Fraction"]
Result = Result[:, 0] # Heart rate here

## EMULATION
idx = np.random.choice(len(Result), size=30000, replace=False)
X = X[idx,:]
Result = Result[idx]


# compare emulators
ae = AutoEmulate()
ae.setup(X, Result,  param_search=True, param_search_iters=30, n_jobs=-1, models=["rbf"], cross_validator=KFold(n_splits=3))
# ae.setup(X, Result, models=[GaussianProcess])
best_emulator = ae.compare()
ae.summarise_cv()
ae.plot_cv()
ae.plot_cv(style="residual_vs_predicted")


rbf = ae.get_model("RadialBasisFunctions")
ae.evaluate(rbf)
ae.plot_eval(rbf)
rbf_final = ae.refit(rbf)

ae.save(rbf_final, "rbf")


# gp = ae.get_model("GaussianProcess")
# ae.evaluate(gp)
# ae.plot_eval(gp)
# gp_final = ae.refit(gp)

# os.makedirs("gp_final", exist_ok=True)
# joblib.dump(gp_final, "gp_final/GaussianProcess.joblib")

# ae.save(gp_final, "gp_final")