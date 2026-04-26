import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


X_PATH = Path(__file__).with_name("DGSM_500_X_rest_20_10_04.npy")
RESULT_PATH = Path(__file__).with_name("DGSM_500_Result_rest_20_10_04.npy")
# X_PATH = Path(__file__).with_name("DGSM_500_X_rest_50_14_04.npy")
# RESULT_PATH = Path(__file__).with_name("DGSM_500_Result_rest_50_14_04.npy")
OUTPUT_PATH = Path(__file__).with_name("check_DGSM_simulation_20.png")

PARAMETER_INDEX = 0
N_ARROWS_PER_PANEL = 8

output_names = [
    "Heart Rate",
    "LV Systolic Pressure",
    "LV Diastolic Pressure",
    "LV EDV",
    "LV ESV",
    "Max RV Volume",
    "Min RV Volume",
    "Max RV Pressure",
    "Min RV Pressure",
    "Min RA Volume",
    "Max RA Volume",
    "Min RA Pressure A descent",
    "Max RA Pressure A Wave",
    "Max RA Pressure V Wave",
    "Min RA Pressure V descent",
    "Min LA Volume",
    "Max LA Volume",
    "Min LA Pressure A descent",
    "Max LA Pressure A Wave",
    "Max LA Pressure V Wave",
    "Min LA Pressure V descent",
    "LA Pre-Atrial Contraction Volume",
    "RA Pre-Atrial Contraction Volume",
    "Max LV Pressure Deriv",
    "Max RV Pressure Deriv",
    "Tidal Volume",
    "Minute Ventilation",
    "Cardiac Output",
    "PaO2",
    "PaCO2",
    "Percentage Volume Change",
    "Stroke Volume",
    "Ejection Fraction",
]

output_names_reduced = [
    "Heart Rate",
    "LV Systolic Pressure",
    "LV Diastolic Pressure",
    "LV EDV",
    "LV ESV",
    "Max RV Volume",
    "Min RV Volume",
    "Max RV Pressure",
    "Min RV Pressure",
    "Min RA Volume",
    "Max RA Volume",
    "Max RA Pressure A Wave",
    "Max RA Pressure V Wave",
    "Min LA Volume",
    "Max LA Volume",
    "Max LA Pressure A Wave",
    "Max LA Pressure V Wave",
    "LA Pre-Atrial Contraction Volume",
    "RA Pre-Atrial Contraction Volume",
    "Max LV Pressure Deriv",
    "Max RV Pressure Deriv",
    "Tidal Volume",
    "Minute Ventilation",
    "PaO2",
    "PaCO2",
]


def get_arrow_indices(x_base, y_base, x_pert, y_pert, n_arrows):
    if len(x_base) == 0:
        return np.array([], dtype=int)

    x_span = max(np.ptp(np.concatenate([x_base, x_pert])), 1e-12)
    y_span = max(np.ptp(np.concatenate([y_base, y_pert])), 1e-12)
    score = np.sqrt(((x_pert - x_base) / x_span) ** 2 + ((y_pert - y_base) / y_span) ** 2)

    n_keep = min(n_arrows, len(score))
    return np.argsort(score)[-n_keep:]


def main():
    X = np.load(X_PATH)
    Result = np.load(RESULT_PATH)

    D = X.shape[1]
    block_size = D + 1
    n_blocks = X.shape[0] // block_size
    # Find basepoint indices (first row of each block)
    base_idx = np.arange(0, X.shape[0], block_size)
    # Mask: True if basepoint result != 0
    mask_blocks = Result[base_idx, 0] != 0  # check column 0 (e.g. HR); adjust if needed
    # OR: drop block if *any* nan appears in that block
    mask_blocks_nan = np.array([
        np.all(np.isfinite(Result[i:i + block_size]))  # True if block has no nan
        for i in base_idx
    ])

    # Compute variability (std) within each block
    block_std = np.zeros((n_blocks, Result.shape[1]))

    for b, i in enumerate(base_idx):
        block = Result[i:i + block_size]
        block_std[b] = np.nanstd(block, axis=0)

    for b, i in enumerate(base_idx):
        print(
            f"Block {b:4d} | std = {block_std[b, 3]:.4g}"
            # f"STD: {block_std[b]}"
        )

    # Threshold = mean + 3*std of block stds, computed across blocks for each output
    std_mean = np.nanmean(block_std, axis=0)
    std_std = np.nanstd(block_std, axis=0)
    std_thresh = std_mean + 3 * std_std

    # Keep blocks only if ALL output stds are below their respective thresholds
    mask_blocks_std = np.all(block_std <= std_thresh, axis=1)

    HR_col = 0
    mask_blocks_conv = np.array([
        np.all(np.abs(Result[i + 1:i + block_size, HR_col] - Result[i, HR_col]) < 0.03)
        for i in base_idx
    ])

    HR_col = 25
    mask_blocks_conv_tidal = np.array([
        np.all(np.abs(Result[i + 1:i + block_size, HR_col] - Result[i, HR_col]) < 0.03)
        for i in base_idx
    ])

    mask_blocks = mask_blocks & mask_blocks_nan & mask_blocks_conv & mask_blocks_std #& mask_blocks_conv_tidal  # & mask_blocks_std # & mask_blocks_E_rs & mask_blocks_R_rs # & mask_blocks_std
    print(np.count_nonzero(mask_blocks))
    # Expand mask to all rows in a block
    mask_full = np.repeat(mask_blocks, block_size)

    # Filter arrays
    X = X[mask_full]
    Result = Result[mask_full]

    reduced_indices = [output_names.index(name) for name in output_names_reduced]
    if max(reduced_indices) >= Result.shape[1]:
        raise ValueError(
            f"Result has {Result.shape[1]} columns, but the reduced target list "
            f"needs column {max(reduced_indices)}."
        )

    block_size = X.shape[1] + 1
    n_complete_blocks = X.shape[0] // block_size

    base_idx = np.arange(0, n_complete_blocks * block_size, block_size)
    pert_idx = base_idx + PARAMETER_INDEX + 1

    x_base = X[base_idx, PARAMETER_INDEX]
    x_pert = X[pert_idx, PARAMETER_INDEX]

    ncols = 5
    nrows = math.ceil(len(output_names_reduced) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 18), constrained_layout=True)
    axes = np.asarray(axes).ravel()

    for ax, target_name, target_idx in zip(axes, output_names_reduced, reduced_indices):
        y_base = Result[base_idx, target_idx]
        y_pert = Result[pert_idx, target_idx]

        valid = (
            np.isfinite(x_base)
            & np.isfinite(x_pert)
            & np.isfinite(y_base)
            & np.isfinite(y_pert)
        )

        xb = x_base[valid]
        xp = x_pert[valid]
        yb = y_base[valid]
        yp = y_pert[valid]

        ax.scatter(xb, yb, s=26, alpha=0.65, color="tab:orange", label="Base point")
        ax.scatter(xp, yp, s=26, alpha=0.65, color="tab:blue", label="Perturbed point")

        for idx in get_arrow_indices(xb, yb, xp, yp, N_ARROWS_PER_PANEL):
            ax.annotate(
                "",
                xy=(xp[idx], yp[idx]),
                xytext=(xb[idx], yb[idx]),
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.4, alpha=0.85),
            )

        ax.set_title(target_name, fontsize=10)
        ax.grid(True, alpha=0.35)

    for ax in axes[len(output_names_reduced):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.supxlabel(f"Parameter {PARAMETER_INDEX + 1} (X[:, {PARAMETER_INDEX}])", fontsize=14)
    fig.supylabel("Target value", fontsize=14)
    fig.suptitle("First-Parameter DGSM Scatter for Reduced Targets", fontsize=18)

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
