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
matplotlib.use('Agg')  # non-interactive backend
import numpy as np
from autoemulate import AutoEmulate
# A = np.load(r'C:\Users\vanes\Downloads\Result_DGSM_chunked1.npy')
# AA = np.load(r'C:\Users\vanes\Downloads\Result_DGSM_chunked.npy')

from sklearn.model_selection import KFold

X_all = np.load('DGSM_filtered_LHCS_500000_X_sample_HR_Plv_Prv_Vlv_Vrv_rest.npy')[:8000,:] # size [N, 299]
Result_all = np.load('Result_8000.npy')

mask = Result_all[:,0] != 0

X = X_all[mask, :]
Result = Result_all[mask, :]


Stroke_Volume = Result[:, 3] - Result[:, 4]
Ejection_fraction = (Stroke_Volume / Result[:, 3]) * 100

Result = np.column_stack((Result, Stroke_Volume))
Result_next = np.column_stack((Result, Ejection_fraction))


# choose which results (column) to look at
Result_cols = ["Heart Rate", "Systolic Pressure", "Diastolic Pressure", "EDV", "ESV", "Max RV Volume", "Min RV Volume",
               "Max RV Pressure", "Min RV Pressure", "Stroke Volume", "Ejection Fraction"]



# # P_dia
# Result = Result_next[:, 2] # Heart rate here
#
# ## EMULATION
# idx = np.random.choice(len(Result), size=30000, replace=False)
# X = X[idx,:]
# Result = Result[idx]
#
#
# # compare emulators
# ae = AutoEmulate(X, Result, log_level="error", models=["rbf"])
# # ae.setup(X, Result,  param_search=True, param_search_iters=30, n_jobs=-1, models=["rbf"], cross_validator=KFold(n_splits=3))
# # ae.setup(X, Result, n_jobs=-1, models=["rbf"])
# ae.summarise()
# best = ae.best_result()
# print(best.params)
#
# # ae.save(best, "rbf_new")
# os.makedirs("rbf_p_dia", exist_ok=True)
# joblib.dump(best, "rbf_p_dia/RBF.joblib")
#
# fig = ae.plot(best, fname="best_model_plot_all_p_dia.png")
# # fig = ae.plot_one(best, input_index=0, output_index=0, fname="best_model_plot_one_p_dia.png")





# # ESV
# Result = Result_next[:, 4] # Heart rate here
#
# ## EMULATION
# idx = np.random.choice(len(Result), size=30000, replace=False)
# X = X_all[idx,:]
# Result = Result[idx]
#
#
# # compare emulators
# ae = AutoEmulate(X, Result, log_level="error", models=["rbf"])
# # ae.setup(X, Result,  param_search=True, param_search_iters=30, n_jobs=-1, models=["rbf"], cross_validator=KFold(n_splits=3))
# # ae.setup(X, Result, n_jobs=-1, models=["rbf"])
# ae.summarise()
# best = ae.best_result()
# print(best.params)
#
# # ae.save(best, "rbf_new")
# os.makedirs("rbf_ESV", exist_ok=True)
# joblib.dump(best, "rbf_ESV/RBF.joblib")
#
# fig = ae.plot(best, fname="best_model_plot_all_ESV.png")
# fig = ae.plot_one(best, input_index=0, output_index=0, fname="best_model_plot_one_ESV.png")



# Max RV Volume
Result = Result_next[:, 5] # Heart rate here

## EMULATION
idx = np.random.choice(len(Result), size=30000, replace=False)
X = X[idx,:]
Result = Result[idx]


# compare emulators
ae = AutoEmulate(X, Result, log_level="error", models=["rbf"])
# ae.setup(X, Result,  param_search=True, param_search_iters=30, n_jobs=-1, models=["rbf"], cross_validator=KFold(n_splits=3))
# ae.setup(X, Result, n_jobs=-1, models=["rbf"])
ae.summarise()
best = ae.best_result()
print(best.params)

# ae.save(best, "rbf_new")
os.makedirs("rbf_max_RV_volume", exist_ok=True)
joblib.dump(best, "rbf_max_RV_volume/RBF.joblib")

fig = ae.plot(best, fname="best_model_plot_all_max_RV_volume.png")





# Min RV Volume
Result = Result_next[:, 6] # Heart rate here

## EMULATION
idx = np.random.choice(len(Result), size=30000, replace=False)
X = X_all[idx,:]
Result = Result[idx]


# compare emulators
ae = AutoEmulate(X, Result, log_level="error", models=["rbf"])
# ae.setup(X, Result,  param_search=True, param_search_iters=30, n_jobs=-1, models=["rbf"], cross_validator=KFold(n_splits=3))
# ae.setup(X, Result, n_jobs=-1, models=["rbf"])
ae.summarise()
best = ae.best_result()
print(best.params)

# ae.save(best, "rbf_new")
os.makedirs("rbf_min_RV_volume", exist_ok=True)
joblib.dump(best, "rbf_min_RV_volume/RBF.joblib")

fig = ae.plot(best, fname="best_model_plot_all_min_RV_volume.png")





# Max RV Pressure
Result = Result_next[:, 7] # Heart rate here

## EMULATION
idx = np.random.choice(len(Result), size=30000, replace=False)
X = X_all[idx,:]
Result = Result[idx]


# compare emulators
ae = AutoEmulate(X, Result, log_level="error", models=["rbf"], device="cuda")
# ae.setup(X, Result,  param_search=True, param_search_iters=30, n_jobs=-1, models=["rbf"], cross_validator=KFold(n_splits=3))
# ae.setup(X, Result, n_jobs=-1, models=["rbf"])
ae.summarise()
best = ae.best_result()
print(best.params)

# ae.save(best, "rbf_new")
os.makedirs("rbf_max_RV_pressure", exist_ok=True)
joblib.dump(best, "rbf_max_RV_pressure/RBF.joblib")

fig = ae.plot(best, fname="best_model_plot_all_max_RV_pressure.png")






# Min RV Pressure
Result = Result_next[:, 8] # Heart rate here

## EMULATION
idx = np.random.choice(len(Result), size=30000, replace=False)
X = X_all[idx,:]
Result = Result[idx]


# compare emulators
ae = AutoEmulate(X, Result, log_level="error", models=["rbf"])
# ae.setup(X, Result,  param_search=True, param_search_iters=30, n_jobs=-1, models=["rbf"], cross_validator=KFold(n_splits=3))
# ae.setup(X, Result, n_jobs=-1, models=["rbf"])
ae.summarise()
best = ae.best_result()
print(best.params)

# ae.save(best, "rbf_new")
os.makedirs("rbf_min_RV_pressure", exist_ok=True)
joblib.dump(best, "rbf_min_RV_pressure/RBF.joblib")

fig = ae.plot(best, fname="best_model_plot_all_min_RV_pressure.png")







# Stroke Volume
Result = Result_next[:, 9] # Heart rate here

## EMULATION
idx = np.random.choice(len(Result), size=30000, replace=False)
X = X_all[idx,:]
Result = Result[idx]


# compare emulators
ae = AutoEmulate(X, Result, log_level="error", models=["rbf"])
# ae.setup(X, Result,  param_search=True, param_search_iters=30, n_jobs=-1, models=["rbf"], cross_validator=KFold(n_splits=3))
# ae.setup(X, Result, n_jobs=-1, models=["rbf"])
ae.summarise()
best = ae.best_result()
print(best.params)

# ae.save(best, "rbf_new")
os.makedirs("rbf_stroke_v", exist_ok=True)
joblib.dump(best, "rbf_stroke_v/RBF.joblib")

fig = ae.plot(best, fname="best_model_plot_all_stroke_v.png")






# eject
Result = Result_next[:, 10] # Heart rate here

## EMULATION
idx = np.random.choice(len(Result), size=30000, replace=False)
X = X_all[idx,:]
Result = Result[idx]


# compare emulators
ae = AutoEmulate(X, Result, log_level="error", models=["rbf"])
# ae.setup(X, Result,  param_search=True, param_search_iters=30, n_jobs=-1, models=["rbf"], cross_validator=KFold(n_splits=3))
# ae.setup(X, Result, n_jobs=-1, models=["rbf"])
ae.summarise()
best = ae.best_result()
print(best.params)

# ae.save(best, "rbf_new")
os.makedirs("rbf_eject", exist_ok=True)
joblib.dump(best, "rbf_eject/RBF.joblib")

fig = ae.plot(best, fname="best_model_plot_all_eject.png")



















# fig = ae.plot_cv()
# fig = ae.plot_cv(style="residual_vs_predicted")
# fig.savefig("cv_plot.png", dpi=300, bbox_inches="tight")  # save figure

# rbf = ae.best_result()
# fig.savefig("eval_plot.png", dpi=300, bbox_inches="tight")  # save figure
#
# rbf_final = ae.refit(rbf)

# gp = ae.get_model("GaussianProcess")
# ae.evaluate(gp)
# ae.plot_eval(gp)
# gp_final = ae.refit(gp)

# os.makedirs("gp_final", exist_ok=True)
# joblib.dump(gp_final, "gp_final/GaussianProcess.joblib")

# ae.save(gp_final, "gp_final")