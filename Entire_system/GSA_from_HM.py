import os
import joblib
import pandas as pd
import seaborn as sns
import torch
from SALib import ProblemSpec
from History_matching_function_new import HistoryMatchingWorkflow
from AutoEmulate_Simulator import Cardiopulmonary
from SALib.analyze import sobol
from sobol_analyze_NIMP import analyze_NIMP
from scipy.stats import norm
from SALib.util import scale_samples
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import logging
logging.getLogger("multiprocessing.resource_tracker").setLevel(logging.CRITICAL)

X_all = np.load(f'NROY_Points_rest_20.npy', allow_pickle=True)
Param_ranges = np.load(f'NROY_Params_rest_20.npy', allow_pickle=True).item()
Emulator_dir = "Emulator_Paper_1_90_same_1000"
calc_second_order = False

########################## Permute to form Sobol samples
subset_vars = [name for name in Param_ranges.keys() if Param_ranges[name][0] != Param_ranges[name][1]]

# Get all indices corresponding to subset_vars
subset_idx = np.array([i for i, name in enumerate(list(Param_ranges.keys())) if Param_ranges[name][0] != Param_ranges[name][1]])
X_subset = X_all[:, subset_idx]
subset_bounds = [Param_ranges[name] for name in subset_vars]

sp_subset = ProblemSpec({
    'names': subset_vars,
    'bounds': subset_bounds
})

subset_param_ranges: dict[str, tuple[float, float]] = {
    str(name): (float(b[0]), float(b[1]))
    for name, b in zip(sp_subset["names"], sp_subset["bounds"])
}

N = X_subset.shape[0] // 2
D = len(subset_idx)

if calc_second_order:
    sample_size = N * (2 * D + 2)
else:
    sample_size = (D + 2) * N

skip_values = 0

base_sequence = np.zeros((N + skip_values, 2 * D),dtype=float)
base_sequence[:,:D] = X_subset[:N,:]
base_sequence[:,D:] = X_subset[N:(N*2),:]

saltelli_sequence = np.zeros([sample_size, D])

index = 0
for i in range(N):
    # Copy matrix "A"
    saltelli_sequence[index, :] = base_sequence[i, :D]
    index += 1

    # 2. Cross-sample hybrids (A with one column from B)
    for k in range(D):
        saltelli_sequence[index, :] = base_sequence[i, :D]
        saltelli_sequence[index, k] = base_sequence[i, k + D]
        index += 1

    # Copy matrix "B"
    saltelli_sequence[index, :] = base_sequence[i, D:]
    index += 1

    # Cross-sample elements of "A" into "B"
    # Only needed if you're doing second-order indices (true by default)
    if calc_second_order:
        for k in range(D):
            # Start with all columns from B
            saltelli_sequence[index, :] = base_sequence[i, D:]
            # Replace the k-th column with A
            saltelli_sequence[index, k] = base_sequence[i, k]
            index += 1

X = torch.from_numpy(saltelli_sequence.astype(np.float32))



########################## Predict outputs from saved emulators and calculate implausibility
output_names = [
    "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV",
    "ESV", "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure",
    "Min_RV_Pressure", "Min_RA_Volume", "Max_RA_Volume", "Max_RA_Pressure_Atrial_contraction",
    "Max_RA_Pressure_Tricuspid_Opening", "Min_LA_Volume", "Max_LA_Volume", "Max_LA_Pressure_Atrial_contraction",
    "Max_LA_Pressure_Mitral_Opening", "LA_Contraction_Volume_diff", "RA_Contraction_Volume_diff", "LV_Pressure_Deriv",
    "RV_Pressure_Deriv", "Tidal_Volume", "Minute_Ventilation", "PaO2",
    "PaCO2"]

observation = {"Heart Rate": (1.1, 0.01), "Systolic Pressure": (105, 25), "Diastolic Pressure": (70, 9),
 "EDV": (163, 529), "ESV": (50, 100), "Max RV Volume": (186, 441),
 "Min RV Volume": (52, 81), "Max RV Pressure": (24, 4), "Min RV Pressure": (2, 1),
 "Min RA Volume": (45, 225), "Max RA Volume": (93, 256), "Max RA Pressure Atrial contraction": (7, 4),
 "Max RA Pressure Tricuspid Opening": (7, 4), "Min LA Volume": (45, 225), "Max LA Volume": (72, 144),
 "Max LA Pressure Atrial contraction": (7, 4), "Max LA Pressure Mitral Opening": (7, 4), "LA Contraction Volume diff": (10, 4),
 "RA Contraction Volume diff": (10, 4), "LV Pressure Deriv": (1600, 93025), "RV Pressure Deriv": (500, 22500),
 "Tidal Volume": (0.5, 0.01), "Minute Ventilation": (6.5, 0.25), "PaO2": (95, 20.25), "PaCO2": (40, 4)}

models = {}
for name in output_names:
    folder = name
    path1 = os.path.join(Emulator_dir, folder, f"GaussianProcessMatern32_{name}_best.joblib")
    models[name] = joblib.load(path1)

# new way parallelised by output
n_jobs = 25
def predict_one_output(name, X):
    target_emulator = models[name].model
    mean, var = target_emulator.predict_mean_and_variance(X)
    return name, mean, var

results = Parallel(n_jobs=n_jobs)(
    delayed(predict_one_output)(name, X) for name in output_names)

means = {name: mean for name, mean, var in results}
variances = {name: var for name, mean, var in results}

# separate chunk size
# chunk_size = 20000
# means = {}
# variances = {}
# n_rows = X.shape[0]
#
# for name in output_names:
#     target_emulator = models[name].model
#     mean_chunks = []
#     var_chunks = []
#
#     for start in range(0, n_rows, chunk_size):
#         end = min(start + chunk_size, n_rows)
#         x_chunk = X[start:end]
#
#         mean_chunk, var_chunk = target_emulator.predict_mean_and_variance(x_chunk)
#         mean_chunks.append(mean_chunk)
#         var_chunks.append(var_chunk)
#
#     means[name] = torch.cat(mean_chunks, dim=0)
#     variances[name] = torch.cat(var_chunks, dim=0)

# # # old way non parallelised, use when debugging
# means = {}
# variances = {}
# for name in output_names:
#     target_emulator = models[name].model
#     means[name], variances[name] = target_emulator.predict_mean_and_variance(X)

mean_tensor = torch.cat([means[name].reshape(-1, 1) for name in output_names], dim=1)
var_tensor = torch.cat([variances[name].reshape(-1, 1) for name in output_names], dim=1)

hmw = HistoryMatchingWorkflow(
    simulator=Cardiopulmonary(param_ranges=Param_ranges, output_names=output_names),
    result=models["Heart_Rate"],
    observations=observation,
    threshold=3.0,
    random_seed=42,
    calibration_params=subset_vars,
)
Implaus = hmw.calculate_implausibility(mean_tensor, var_tensor)
Implaus = Implaus.detach().cpu().numpy()
# top = np.partition(Implaus.ravel(), -50)[-50:]
# top = np.sort(top)[::-1]
# print(top)

########################## Remove entire A/B if even a single permutation is outside an implausibility of 3
block_length = 2 * D + 2 if calc_second_order else D + 2
Implaus_max = Implaus.max(axis=1)   # shape: (sample_size,)
blocks = Implaus_max.reshape(N, block_length)
valid_mask = np.all(blocks <= 3.08, axis=1)
valid_indices = np.where(valid_mask)[0]

# Create a mask over all rows
row_mask = np.repeat(valid_mask, block_length)

# Filter everything in one go
filtered_saltelli = saltelli_sequence[row_mask]
filtered_Implaus = Implaus[row_mask]
filtered_Result = mean_tensor[row_mask]

index = 0
for i in valid_indices:
    start = i * block_length
    end = start + block_length
    filtered_saltelli[index:index + block_length, :] = saltelli_sequence[start:end, :]
    filtered_Implaus[index:index + block_length] = Implaus[start:end]
    filtered_Result[index:index + block_length] = mean_tensor[start:end]
    index += block_length

print(f"Number of base A/B blocks remaining: {len(valid_indices)}")
print(f"Number of base A/B blocks originally: {N}")


import math

# ==========================
# KDE plots for all outputs
# ==========================
n_outputs = len(output_names)
n_cols = 5
n_rows = math.ceil(n_outputs / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
axes = np.atleast_1d(axes).ravel()

for i, name in enumerate(output_names):
    ax = axes[i]
    vals = mean_tensor[:, i].detach().cpu().numpy()
    sns.kdeplot(vals, fill=True, ax=ax)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")

for j in range(n_outputs, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("All_outputs_KDE.png", dpi=300, bbox_inches="tight")
plt.close()


################# at least 1% combine up to 90%
def select_important_parameters(df, index_col, err_col=None, min_frac=0.01, coverage=0.90):
    """
    Keep parameters that:
    1) each contribute at least `min_frac` of total sensitivity
    2) together account for up to `coverage` of total sensitivity

    df must already contain Sobol indices for one output.
    """
    out = df.copy()

    # avoid negative numerical artefacts affecting the fractions
    vals = np.asarray(out[index_col].values, dtype=float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.clip(vals, 0.0, None)

    out[index_col] = vals
    out = out.sort_values(index_col, ascending=False)

    total = out[index_col].sum()

    # fallback: if everything is zero, just keep top 10
    if total <= 0:
        return out.head(min(10, len(out)))

    out["fraction"] = out[index_col] / total
    out["cum_fraction"] = out["fraction"].cumsum()

    # keep >=1% contributors
    keep = out["fraction"] >= min_frac

    # keep until cumulative reaches 90%
    # include the first parameter that crosses 90%
    cross_idx = np.searchsorted(out["cum_fraction"].values, coverage, side="left")
    keep.iloc[:cross_idx + 1] = True

    filtered = out.loc[keep].copy()

    # in case min_frac removes everything except weird edge cases
    if filtered.empty:
        filtered = out.head(min(10, len(out))).copy()

    return filtered


# ==========================
# Sobol analysis for all outputs
# ==========================
param_names = sp_subset["names"]
conf_level = 0.95
z = norm.ppf(0.5 + conf_level / 2)

sobol_results = {}

for i, out_name in enumerate(output_names):
    Y = mean_tensor[:, i].detach().cpu().numpy().copy()

    S = analyze_NIMP(
        sp_subset,
        Y,
        calc_second_order=calc_second_order,
        print_to_console=False
    )

    T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S)

    total = pd.DataFrame({
        "Parameter": param_names,
        "ST": np.asarray(T_Si["ST"], dtype=float),
        "ST_std": np.asarray(T_Si["ST_conf"], dtype=float) / z,
        "S1": np.asarray(first_Si["S1"], dtype=float),
        "S1_std": np.asarray(first_Si["S1_conf"], dtype=float) / z,
    }).set_index("Parameter")

    sobol_results[out_name] = {
        "table": total,
        "S_raw": S,
        "second_Si": second_Si,
    }


# ==========================
# ST ranked barplots for all outputs
# ==========================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows))
axes = np.atleast_1d(axes).ravel()

for i, out_name in enumerate(output_names):
    ax = axes[i]
    # ranked = sobol_results[out_name]["table"].sort_values("ST", ascending=False)
    ranked = select_important_parameters(
        sobol_results[out_name]["table"],
        index_col="ST",
        err_col="ST_std",
        min_frac=0.01,
        coverage=0.90
    )

    ax.barh(
        ranked.index,
        ranked["ST"].values,
        xerr=ranked["ST_std"].values,
        edgecolor="k"
    )
    ax.invert_yaxis()
    ax.set_title(f"{out_name} - ST", fontsize=10)
    ax.set_xlabel("Total-order index")
    ax.tick_params(axis="y", labelsize=8)

for j in range(n_outputs, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("Sobol_ST_ranked_all_outputs.png", dpi=300, bbox_inches="tight")
plt.close()


# ==========================
# S1 ranked barplots for all outputs
# ==========================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows))
axes = np.atleast_1d(axes).ravel()

for i, out_name in enumerate(output_names):
    ax = axes[i]
    ranked = select_important_parameters(
        sobol_results[out_name]["table"],
        index_col="S1",
        err_col="S1_std",
        min_frac=0.01,
        coverage=0.90
    )
    ax.barh(
        ranked.index,
        ranked["S1"].values,
        xerr=ranked["S1_std"].values,
        edgecolor="k"
    )
    ax.invert_yaxis()
    ax.set_title(f"{out_name} - S1", fontsize=10)
    ax.set_xlabel("First-order index")
    ax.tick_params(axis="y", labelsize=8)

for j in range(n_outputs, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("Sobol_S1_ranked_all_outputs.png", dpi=300, bbox_inches="tight")
plt.close()


# ==========================
# Optional: save CSV per output
# ==========================
os.makedirs("Sobol_tables", exist_ok=True)
for out_name in output_names:
    sobol_results[out_name]["table"].sort_values("ST", ascending=False).to_csv(
        os.path.join("Sobol_tables", f"{out_name}_sobol_indices.csv"),
        index=True
    )


# ==========================
# Optional second-order plots for all outputs
# ==========================
if calc_second_order:
    os.makedirs("Sobol_S2_plots", exist_ok=True)

    param_pairs = [
        (param_names[i], param_names[j])
        for i in range(D) for j in range(i + 1, D)
    ]

    for out_name in output_names:
        second_Si = sobol_results[out_name]["second_Si"]

        S2_df = pd.DataFrame({
            "Parameter_pair": [" & ".join(pair) for pair in param_pairs],
            "S2": np.asarray(second_Si["S2"], dtype=float).flatten(),
            "S2_std": np.asarray(second_Si["S2_conf"], dtype=float).flatten() / z
        }).sort_values("S2", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 16))
        ax.barh(
            S2_df["Parameter_pair"],
            S2_df["S2"],
            xerr=S2_df["S2_std"],
            edgecolor="k"
        )
        ax.invert_yaxis()
        ax.set_title(f"{out_name} - S2", fontsize=14)
        ax.set_xlabel("Second-order index")
        plt.tight_layout()
        plt.savefig(
            os.path.join("Sobol_S2_plots", f"{out_name}_S2_ranked.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()