import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

warnings.filterwarnings(
    "ignore",
    message="use_inf_as_na option is deprecated",
    category=FutureWarning,
)

SCRIPT_DIR = Path(__file__).resolve().parent

ALL_OUTPUT_NAMES = [
    "Heart Rate", "Systolic Pressure", "Diastolic Pressure", "EDV", "ESV",
    "Max RV Volume", "Min RV Volume", "Max RV Pressure", "Min RV Pressure",
    "Min RA Volume", "Max RA Volume", "Min RA Pressure A descent",
    "Max RA Pressure Atrial contraction", "Max RA Pressure Tricuspid Opening",
    "Min RA Pressure V descent", "Min LA Volume", "Max LA Volume",
    "Min LA Pressure A descent", "Max LA Pressure Atrial contraction",
    "Max LA Pressure Mitral Opening", "Min LA Pressure V descent",
    "LA Contraction Volume diff", "RA Contraction Volume diff",
    "LV Pressure Deriv", "RV Pressure Deriv", "Tidal Volume",
    "Minute Ventilation", "Cardiac Output", "PaO2", "PaCO2",
    "Percentage Volume Change",
]

EMULATED_OUTPUT_NAMES = [
    "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV",
    "ESV", "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure",
    "Min_RV_Pressure", "Min_RA_Volume", "Max_RA_Volume",
    "Max_RA_Pressure_Atrial_contraction",
    "Max_RA_Pressure_Tricuspid_Opening", "Min_LA_Volume", "Max_LA_Volume",
    "Max_LA_Pressure_Atrial_contraction", "Max_LA_Pressure_Mitral_Opening",
    "LA_Contraction_Volume_diff", "RA_Contraction_Volume_diff",
    "LV_Pressure_Deriv", "RV_Pressure_Deriv", "Tidal_Volume",
    "Minute_Ventilation", "PaO2", "PaCO2",
]

REMOVED_OUTPUT_INDICES = [11, 14, 17, 20, 27, 30]
PLOTTED_OUTPUT_NAMES = [
    name for idx, name in enumerate(ALL_OUTPUT_NAMES)
    if idx not in REMOVED_OUTPUT_INDICES
]

SUBSET_IDX = [
    0, 1, 2, 3, 11, 13, 14, 15, 18, 24, 37, 42, 43, 45, 46, 47, 48, 49,
    50, 51, 52, 54, 55, 56, 58, 59, 60, 64, 68, 79, 81, 93, 94, 95, 120,
    124, 125, 127, 128, 137, 138, 139, 146, 158, 161, 162, 174, 179, 184,
    191, 195, 197, 201, 202, 203, 204, 215, 216, 218, 246, 247, 263, 264,
    265, 266, 268, 269, 270, 271,
]


def load_initial_results() -> np.ndarray:
    result_initial = np.load(SCRIPT_DIR / "LHCS_Result_20.npy")

    if result_initial.shape[1] == len(ALL_OUTPUT_NAMES):
        result_initial = np.delete(result_initial, REMOVED_OUTPUT_INDICES, axis=1)
    elif result_initial.shape[1] != len(PLOTTED_OUTPUT_NAMES):
        raise ValueError(
            f"Unexpected LHCS_Result_20.npy shape {result_initial.shape}; "
            f"expected 31 or 25 output columns."
        )

    return result_initial


def resolve_run_paths() -> tuple[Path, Path, Path, Path, Path]:
    candidates = [
        # (
        #     SCRIPT_DIR / "4wave_pre_A_calib",
        #     SCRIPT_DIR / "4wave_pre_A_calib" / "Emulator_wave_4wave",
        #     "4wave",
        # ),
        (
            SCRIPT_DIR / "three_implaus_pre_A_calib",
            SCRIPT_DIR / "Emulator_wave_1wave",
            "12_4",
        ),
    ]

    for root_folder, emulator_root, suffix in candidates:
        implaus_path = root_folder / f"NROY_Implaus_rest_20_{suffix}.npy"
        test_param_path = root_folder / f"test_param_rest_20_{suffix}.npy"
        if implaus_path.exists() and test_param_path.exists() and emulator_root.exists():
            kde_file = root_folder / "KDE_prior_posterior.png"
            return root_folder, emulator_root, implaus_path, test_param_path, kde_file

    raise FileNotFoundError(
        "Could not find a matching calibration folder, test parameter file, "
        "and emulator directory under Entire_system."
    )


def main() -> None:
    if len(PLOTTED_OUTPUT_NAMES) != len(EMULATED_OUTPUT_NAMES):
        raise ValueError("Output name lists are misaligned.")

    _, emulator_root, implaus_path, test_param_path, kde_file = resolve_run_paths()
    implaus = np.load(implaus_path)
    x_all = np.load(test_param_path)

    mask = np.all(implaus < 3, axis=1)
    x_subset = torch.from_numpy(x_all[mask][:, SUBSET_IDX]).float()

    calibrated_columns = []
    kept_initial_cols = []
    kept_output_names = []

    for col_idx, (plot_name, emulator_name) in enumerate(
        zip(PLOTTED_OUTPUT_NAMES, EMULATED_OUTPUT_NAMES)
    ):
        emu_path = (
            emulator_root
            / emulator_name
            / f"GaussianProcessMatern32_{emulator_name}_best.joblib"
        )
        if not emu_path.exists():
            print(f"[skip] missing {emu_path}")
            continue

        emulator = joblib.load(emu_path)
        mean_pred, _ = emulator.predict_mean_and_variance(x_subset)
        mean_pred = torch.as_tensor(mean_pred).detach().cpu().reshape(-1, 1).numpy()

        calibrated_columns.append(mean_pred)
        kept_initial_cols.append(col_idx)
        kept_output_names.append(plot_name)

    if not calibrated_columns:
        raise FileNotFoundError(
            f"No emulator predictions were loaded from {emulator_root}."
        )

    result_calibrated = np.hstack(calibrated_columns)
    result_initial = load_initial_results()[:, kept_initial_cols]

    n_outputs = len(kept_output_names)
    n_cols = 5
    n_rows = math.ceil(n_outputs / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows)
    )
    axes = np.atleast_1d(axes).ravel()

    for i, name in enumerate(kept_output_names):
        ax = axes[i]

        vals_initial = np.asarray(result_initial[:, i])
        vals_calibrated = np.asarray(result_calibrated[:, i])

        vals_initial = vals_initial[np.isfinite(vals_initial)]
        vals_calibrated = vals_calibrated[np.isfinite(vals_calibrated)]

        sns.kdeplot(
            vals_initial,
            ax=ax,
            label="Initial",
            fill=True,
            alpha=0.3,
            linewidth=1.5,
        )

        sns.kdeplot(
            vals_calibrated,
            ax=ax,
            label="Calibrated",
            fill=True,
            alpha=0.3,
            linewidth=1.5,
        )

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for j in range(n_outputs, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(kde_file, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
