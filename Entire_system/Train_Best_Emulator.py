import os
import joblib
# import pandas as pd
# import seaborn as sns
# import torch
from SALib import ProblemSpec
from gpytorch.likelihoods import MultitaskGaussianLikelihood

# from SALib.plotting.bar import plot as barplot
# from SALib.analyze import sobol
# from SALib.analyze.sobol import analyze
# # from SALib.sample import saltelli
# from SALib.sample.sobol import sample
# import matplotlib.pyplot as plt
import numpy as np
from autoemulate import AutoEmulate
import warnings
# Ignore only this specific FutureWarning from pandas
warnings.filterwarnings(
    "ignore",
    message=".*use_inf_as_na option is deprecated.*",
    category=FutureWarning
)
from sklearn.model_selection import KFold

# A = AutoEmulate.list_emulators()
# print(A)
# GaussianProcess1 = joblib.load("GaussianProcessMatern32_5000_rest_no_p_thor.joblib")
# print(GaussianProcess1.r2_test)
#
# GaussianProcess1 = joblib.load("GaussianProcessMatern32_10000_rest_no_p_thor.joblib")
# print(GaussianProcess1.r2_test)

# change
size = 1000

# change
X_all = np.load(f'LHCS_20000_X_rest_no_Pthor_Vtot_22_01_2026.npy')
Result_all = np.load('LHCS_20000_Result_rest_no_Pthor.npy')

mask = Result_all[:,0] != 0
X = X_all[mask, :]
Result_all = Result_all[mask]

Result_all = np.delete(Result_all, 27, axis=1)


# try and see if emulator is better with or without filtering
# get the mean of the column
col_mean = Result_all.mean(axis=0)
col_std = Result_all.std(axis=0)
# 3 std to remove outliers
mask = (Result_all >= (col_mean - 3*col_std)) & (Result_all <= (col_mean + 3*col_std))
row_mask = mask.all(axis=1)
X = X[row_mask, :]
Result = Result_all[row_mask]

mask = np.ptp(X, axis=0) != 0  # ptp = max - min, 0 means all values identical
X = X[:, mask]

output_names = [
    "Heart Rate", "Systolic Pressure", "Diastolic Pressure", "EDV", "ESV",
    "Max RV Volume", "Min RV Volume", "Max RV Pressure", "Min RV Pressure",
    "Min RA Volume", "Max RA Volume", "Min RA Pressure A descent", "Max RA Pressure Atrial contraction",
    "Max RA Pressure Tricuspid Opening", "Min RA Pressure V descent",
    "Min LA Volume", "Max LA Volume", "Min LA Pressure A descent", "Max LA Pressure Atrial contraction",
    "Max LA Pressure Tricuspid Opening", "Min LA Pressure V descent",
    "LA EDV", "RA EDV", "LV Pressure Deriv", "RV Pressure Deriv", "Tidal Volume", "Minute Ventilation",
    # "Cardiac Output",
    "PaO2", "PaCO2", "Percentage Volume Change",
#    "Stroke Volume", "Ejection Fraction"
]
#
# rows, cols = 6, 6
# fig, axes = plt.subplots(rows, cols, figsize=(18, 15))
# axes = axes.flatten()
# for i, ax in enumerate(axes):
#     if i < Result.shape[1]:
#         sns.kdeplot(Result[:, i], fill=True, ax=ax)
#         ax.set_title(output_names[i], fontsize=10, pad=1)
#     else:
#         ax.axis("off")
# plt.tight_layout()
# plt.show()


# idx = np.random.choice(len(Result), size=10000, replace=False)
X = X[:size,:]
Result = Result[:size]


# train one emulator at a time
model_name = "GaussianProcessMatern32"
save_root = f"best_{model_name}"
plot_root = f"plots_{model_name}"
os.makedirs(save_root, exist_ok=True)
os.makedirs(plot_root, exist_ok=True)

params = {
    "epochs": 200,
    "lr": 0.1,
    "likelihood_cls": MultitaskGaussianLikelihood,  # class reference is fine (AutoEmulate passes it through)
    "scheduler_cls": None,
    "scheduler_kwargs": {},
}

for j, target_name in enumerate(output_names):
    print("\n" + "=" * 90)
    print(f"Training emulator for target [{j}/{len(output_names)-1}]: {target_name}")
    print("=" * 90)

    # single-output target (N, 1)
    Y = Result[:, j:j+1]
    Y = Y[:, 0]

    ae = AutoEmulate(
        X,
        Y,
        log_level="info",
        models=[model_name],
        # model_params=params,
    )

    ae.summarise()
    best = ae.best_result()

    # print minimal summary (fields depend on AutoEmulate result object)
    r2_test = getattr(best, "r2_test", None)
    rmse_test = getattr(best, "rmse_test", None)
    print(f"Target: {target_name} | R² test: {r2_test} | RMSE test: {rmse_test}")
    print(f"Best model id: {best.id} | model_name: {best.model_name}")
    print(best.params)

    # save model per target
    safe_name = target_name.replace("/", "_").replace(" ", "_")
    outpath = os.path.join(save_root, f"{model_name}_{safe_name}_{size}_rest_no_pthor_vtot_target.joblib")
    joblib.dump(best, outpath)





# # train one emulator for all targetsat the same time
#
# # model_names = ["LightGBM", "SupportVectorMachine", "RandomForest", "MLP", "EnsembleMLP", "EnsembleMLPDropout",
# # "GaussianProcess", "GaussianProcessCorrelated", "GaussianProcessMatern32", "GaussianProcessMatern52", "GaussianProcessRBF"]
# model_names = ["GaussianProcessMatern32"]
# params = {
#     'epochs': 200,
#     'lr': 0.1,
#     'likelihood_cls': MultitaskGaussianLikelihood,  # ← actual class reference
#     'scheduler_cls': None,
#     'scheduler_kwargs': {}
# }
#
# # GaussianProcess_old = joblib.load("best_GaussianProcess/Max_RA_GaussianProcess_10000.joblib")
# # params = GaussianProcess_old.params
#
#
# for model_name in model_names:
#     # change cuda or cpu
#     ae = AutoEmulate(X, Result, log_level="info", models=[model_name], model_params=params)
#     ae.summarise()
#     best = ae.best_result()
#     print(f"Model with id: {best.id} performed best: {best.model_name}")
#     print(best.params)
#
#     os.makedirs(f"best_{model_name}", exist_ok=True)
#     # change
#     joblib.dump(best, f"best_{model_name}/{model_name}_{size}_rest_no_pthor_vtot_target.joblib")
#
#     # change
#     fig = ae.plot(best, fname=f"{model_name}_{size}_target.png")




# EMULATION
# # compare emulators
# ae = AutoEmulate(X, Result, log_level="info", models=["rbf"])
# ae.summarise()
# best = ae.best_result()
# print("Model with id: ", best.id, " performed best: ", best.model_name)
# print(best.params)
#
# # ae.save(best, "rbf_new")
# os.makedirs("best_HR", exist_ok=True)
# joblib.dump(best, "best_HR/HR_rbf_10000.joblib")
#
# fig = ae.plot(best, fname="HR_rbf_10000.png")

# # --- Extract names and bounds from your ProblemSpec
# param_names = np.array(sp["names"])[mask]
# param_bounds = np.array(sp["bounds"])[mask]
# # --- Compute nominal values (midpoint of bounds)
# param_nominal = [(low + high) / 2 for low, high in param_bounds]
# param_min = [low for low, high in param_bounds]
# param_max = [high for low, high in param_bounds]
#
# n_params = X.shape[1]
# chunk_size = 30  # plots per figure
#
# print("Any NaN? ", np.isnan(X).any())
# print("Any +inf? ", np.isposinf(X).any())
# print("Any -inf? ", np.isneginf(X).any())
#
# # --- Filter out constant columns
# valid_indices = [i for i in range(n_params) if not np.all(X[:, i] == X[0, i])]
# # Select the first five valid variables
# subset_indices = valid_indices[:17]
#
# # Create a dataframe with those variables
# X_subset = pd.DataFrame(X[:, subset_indices], columns=[param_names[i] for i in subset_indices])
#
# # Create the pairplot
# sns.pairplot(
#     X_subset,
#     corner=True,        # shows only the lower triangle (less cluttered)
#     diag_kind="kde",    # use KDE plots on the diagonal
#     kind="hist",
#     # plot_kws={"alpha": 0.5, "s": 10, "edgecolor": "none"}  # style for scatter
# )
#
# plt.suptitle("Pairwise Density of First Five Parameters", y=1.02)
# plt.show()
#

# for start in range(0, len(valid_indices), chunk_size):
#     end = min(start + chunk_size, len(valid_indices))
#     subset_indices = valid_indices[start:end]
#
#     ncols = 5
#     nrows = int(np.ceil(len(subset_indices) / ncols))
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows*2.5))
#     axes = axes.flatten()
#
#     for i, param_idx in enumerate(subset_indices):
#         ax = axes[i]
#         sns.kdeplot(X[:, param_idx], fill=True, ax=ax, color="blue", alpha=0.6)
#
#         # Add vertical lines for min, max, nominal
#         ax.axvline(param_min[param_idx], color="red", linestyle="--", label="Min" if i == 0 else "")
#         ax.axvline(param_max[param_idx], color="green", linestyle="--", label="Max" if i == 0 else "")
#         ax.axvline(param_nominal[param_idx], color="black", linestyle="-", label="Nominal" if i == 0 else "")
#
#         ax.set_title(param_names[param_idx], fontsize=8)
#         ax.set_xlabel("")
#         ax.set_ylabel("")
#
#     # Remove unused axes if fewer than nrows*ncols
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])
#
#     # Only add legend once per figure
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="upper right")
#
#     plt.tight_layout()
#     plt.show()


# physiological filters
# hr_mask = (Result[:, 0] < 1.8) & (Result[:, 0] > 0.7)
# hr_mask = Result[:, 0] < 3.67
# X = X[hr_mask, :]
# Result = Result[hr_mask]

# p_sys_mask = (Result[:, 1] < 135) & (Result[:, 1] > 90)
# X = X[p_sys_mask, :]
# Result = Result[p_sys_mask]
#
# p_dia_mask = Result[:, 2] < 100
# X = X[p_dia_mask, :]
# Result = Result[p_dia_mask]
#
# EDV_mask = (Result[:, 3] < 230) & (Result[:, 3] > 95)
# X = X[EDV_mask, :]
# Result = Result[EDV_mask]
#
# ESV_mask = Result[:, 4] > 35
# X = X[ESV_mask, :]
# Result = Result[ESV_mask]
#
# RV_V_max_mask = (Result[:, 5] < 260) & (Result[:, 5] > 100)
# X = X[RV_V_max_mask, :]
# Result = Result[RV_V_max_mask]
#
# RV_V_min_mask = (Result[:, 6] < 135) & (Result[:, 6] > 35)
# X = X[RV_V_min_mask, :]
# Result = Result[RV_V_min_mask]
#
# RV_P_max_mask = (Result[:, 7] < 35) & (Result[:, 7] > 15)
# X = X[RV_P_max_mask, :]
# Result = Result[RV_P_max_mask]
#
# RV_P_min_mask = Result[:, 8] < 8
# X = X[RV_P_min_mask, :]
# Result = Result[RV_P_min_mask]
#
# min_RA_p_mask = Result[:, 11] < 12
# X = X[min_RA_p_mask, :]
# Result = Result[min_RA_p_mask]
#
# max_RA_p_mask = Result[:, 12] < 12
# X = X[max_RA_p_mask, :]
# Result = Result[max_RA_p_mask]
#
# min_LA_p_mask = Result[:, 15] < 12
# X = X[min_LA_p_mask, :]
# Result = Result[min_LA_p_mask]
#
# max_LA_p_mask = Result[:, 16] < 12
# X = X[max_LA_p_mask, :]
# Result = Result[max_LA_p_mask]
#
# EDV_LA_mask = Result[:, 17] < 0.95 * Result[:, 14]
# X = X[EDV_LA_mask, :]
# Result = Result[EDV_LA_mask]
#
# EDV_RA_mask = Result[:, 18] < 0.95 * Result[:, 10]
# X = X[EDV_RA_mask, :]
# Result = Result[EDV_RA_mask]
#
# min_RA_v_mask = Result[:, 9] > 40
# X = X[min_RA_v_mask, :]
# Result = Result[min_RA_v_mask]
#
# min_LA_v_mask = Result[:, 13] > 40
# X = X[min_LA_v_mask, :]
# Result = Result[min_LA_v_mask]
#
# SV_mask = Result[:, 21] < 100
# X = X[SV_mask, :]
# Result = Result[SV_mask]
#
# eject_mask = Result[:, 22] < 80
# X = X[eject_mask, :]
# Result = Result[eject_mask]
#
# no = 10
# print(Result[no,:])
# values = X[no,:]
# Parameters = {name: val for name, val in zip(sp['names'], values)}