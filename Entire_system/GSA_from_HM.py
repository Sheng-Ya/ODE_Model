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
# print(to)

# ########################## Remove entire A/B if even a single permutation is outside an implausibility of 3
# block_length = 2 * D + 2 if calc_second_order else D + 2
# Implaus_max = Implaus.max(axis=1)   # shape: (sample_size,)
# blocks = Implaus_max.reshape(N, block_length)
# valid_mask = np.all(blocks <= 3.2, axis=1)
# valid_indices = np.where(valid_mask)[0]
#
# # Create a mask over all rows
# row_mask = np.repeat(valid_mask, block_length)
#
# # Filter everything in one go
# filtered_saltelli = saltelli_sequence[row_mask]
# filtered_Implaus = Implaus[row_mask]
# filtered_Result = mean_tensor[row_mask]
#
# index = 0
# for i in valid_indices:
#     start = i * block_length
#     end = start + block_length
#     filtered_saltelli[index:index + block_length, :] = saltelli_sequence[start:end, :]
#     filtered_Implaus[index:index + block_length] = Implaus[start:end]
#     filtered_Result[index:index + block_length] = mean_tensor[start:end]
#     index += block_length
#
# print(f"Number of base A/B blocks remaining: {len(valid_indices)}")
# print(f"Number of base A/B blocks originally: {N}")


# # Just HR plot
# fig, ax1 = plt.subplots()
# sns.kdeplot(mean_tensor[:,7], fill=True)
#
# ax1.set_title(f"Filtered RV P")
# ax1.set_xlabel("Value")
# ax1.set_ylabel("Density")
# plt.tight_layout()
# plt.show()

ST = np.zeros((0, D), dtype=float)
S1 = np.zeros((0, D), dtype=float)

ST_std = np.zeros((0, D), dtype=float)
S1_std = np.zeros((0, D), dtype=float)

HR = mean_tensor[:, 7]

S = analyze_NIMP(sp_subset, HR.detach().cpu().numpy().copy(), calc_second_order=calc_second_order, print_to_console=True)
T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S)

ST = np.vstack((ST, T_Si["ST"].reshape(1, -1)))
S1 = np.vstack((S1, first_Si["S1"].reshape(1, -1)))

conf_level = 0.95
z = norm.ppf(0.5 + conf_level / 2)

ST_std = np.vstack((ST_std, T_Si["ST_conf"].reshape(1, -1) / z))
S1_std = np.vstack((S1_std, first_Si["S1_conf"].reshape(1, -1) / z))


# --- Convert to DataFrame for plotting ---
param_names = sp_subset['names']  # assuming this exists
total = pd.DataFrame({
    "Parameter": param_names,
    "ST": ST.flatten(),
    "ST_std": ST_std.flatten(),
    "S1": S1.flatten(),
    "S1_std": S1_std.flatten()
}).set_index("Parameter")

# --- Sort by Total-order sensitivity ---
ranked = total.sort_values("ST", ascending=False)

# ranked.to_csv(f"Plot_abstract/Heart_Rate_sensitivities.csv", index=True)

# --- Bar plot ---
fig, ax = plt.subplots(figsize=(6, 12))
ranked["ST"].plot(kind="barh", xerr=ranked["ST_std"], ax=ax, color="skyblue", edgecolor="k")
ax.invert_yaxis()
# ax.set_xscale("log")
ax.set_title(f"Heart_Rate Sobol Total-Order Sensitivities (Ranked)", fontsize=14)
ax.set_xlabel("Total-order index (ST)")

# Annotate each bar with rank
for i, (name, value) in enumerate(zip(ranked.index, ranked["ST"])):
    ax.text(value * 1.05, i, f"#{i+1}", va="center", ha="left", fontsize=9, color="blue")

plt.tight_layout()
# plt.show()
plt.savefig("Sobol_ST_ranked.png", dpi=300, bbox_inches="tight")
plt.close()


ranked = total.sort_values("S1", ascending=False)

# --- Bar plot ---
fig, ax = plt.subplots(figsize=(6, 12))
ranked["S1"].plot(kind="barh", xerr=ranked["S1_std"], ax=ax, color="skyblue", edgecolor="k")
ax.invert_yaxis()
# ax.set_xscale("log")
ax.set_title("Sobol First-Order Sensitivities (Ranked)", fontsize=14)
ax.set_xlabel("First-order index (S1)")

# Annotate each bar with rank
for i, (name, value) in enumerate(zip(ranked.index, ranked["S1"])):
    ax.text(value * 1.05, i, f"#{i+1}", va="center", ha="left", fontsize=9, color="blue")

plt.tight_layout()
# plt.show()
plt.savefig("Sobol_S1_ranked.png", dpi=300, bbox_inches="tight")
plt.close()

if calc_second_order:
    for j in range(D):
        for k in range(j + 1, D):
            print("%s %s %f %f" % (sp_subset["names"][j], sp_subset["names"][k],
                                   S['S2'][j, k], S['S2_conf'][j, k]))

    S2_list = second_Si["S2"]
    S2_conf_list = second_Si["S2_conf"] / z

    param_pairs = [(sp_subset["names"][i], sp_subset["names"][j]) for i in range(D) for j in range(i + 1, D)]


    S2_df = pd.DataFrame({
        "Parameter_pair": [" & ".join(pair) for pair in param_pairs],
        "S2": np.array(S2_list).flatten(),
        "S2_std": np.array(S2_conf_list).flatten()
    }).sort_values("S2", ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 16))
    S2_df.plot(
        kind="barh",
        x="Parameter_pair",
        y="S2",
        xerr="S2_std",
        ax=ax,
        color="skyblue",
        edgecolor="k"
    )
    ax.invert_yaxis()
    ax.set_title(f"Heart_Rate Sobol Second-Order Sensitivities", fontsize=14)
    ax.set_xlabel("Second-order index (S2)")
    plt.tight_layout()
    plt.show()