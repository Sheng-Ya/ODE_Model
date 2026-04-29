import csv
from SALib import ProblemSpec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

# stopped this plot as I realised that the old wave plot was correct (that was for heart rate and had low implausibility already)
# could try and do like New_param_set_after_rest_cal where it takes the max_implausibility of the row  and plot rather than just one column

# ── Load and filter ──────────────────────────────────────────────
root_folder = "Total_Blood_Volume_Analysis"
end = "V_tot"
Figure_path = f"{root_folder}/implausibility_wave_3_rest.png"
AAAA = np.load(f"{root_folder}/NROY_Implaus_rest_20_{end}.npy")
AAAAA = np.load(f"{root_folder}/test_param_rest_20_{end}.npy")
Param_ranges = np.load(f"{root_folder}/NROY_Params_rest_20_{end}.npy", allow_pickle=True).item()
# keep rows where every implausibility value is < 3
mask = np.all(AAAA < 3, axis=1)
X_calibrated = AAAAA[mask]
implaus_kept = AAAA[mask]

# Random sample of 10000
idx = np.random.choice(X_calibrated.shape[0], size=10000, replace=False)
X_calibrated = X_calibrated[idx]
implaus_kept = implaus_kept[idx][:,0]

# ── Active (non-fixed) parameters ────────────────────────────────
all_param_names = list(Param_ranges.keys())
subset_vars = [name for name in all_param_names if Param_ranges[name][0] != Param_ranges[name][1]]

# ── Plot all active parameters ───────────────────────────────────
n_params = len(subset_vars)
ncols = 4
nrows = int(np.ceil(n_params / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for i, name in enumerate(subset_vars):
    ax = axes[i]
    col_idx = all_param_names.index(name)
    x = X_calibrated[:, col_idx]
    y = implaus_kept

    # Density coloring
    try:
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        density = kde(xy)
    except Exception:
        density = np.ones_like(x)

    sort_idx = density.argsort()
    ax.scatter(x[sort_idx], y[sort_idx], c=density[sort_idx],
               cmap='viridis', s=3, alpha=0.7, rasterized=True)

    # # Highlight the densest point
    # ax.scatter(x[col_idx], y,
    #            c='red', s=250, marker='D', zorder=10,
    #            label='Densest point')

    ax.set_xlabel(name, fontsize=11)
    if i % ncols == 0:
        ax.set_ylabel('Implausibility', fontsize=11)
    else:
        ax.set_ylabel('')
    ax.tick_params(labelsize=9)

# Hide unused subplots
for j in range(n_params, len(axes)):
    axes[j].set_visible(False)

# axes[0].legend(fontsize=9, loc='upper left')
# plt.suptitle('Implausibility vs Parameters — Densest NROY Point (red star)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(Figure_path, dpi=200, bbox_inches='tight')
print("\nPlot saved.")



