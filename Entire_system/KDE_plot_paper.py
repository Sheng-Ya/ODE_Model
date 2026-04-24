import math
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import warnings
import numpy as np
import seaborn as sns

warnings.filterwarnings(
    "ignore",
    message="use_inf_as_na option is deprecated",
    category=FutureWarning
)

output_names = [
    "Heart Rate", "Systolic Pressure", "Diastolic Pressure", "EDV", "ESV",
    "Max RV Volume", "Min RV Volume", "Max RV Pressure", "Min RV Pressure",
    "Min RA Volume", "Max RA Volume", "Min RA Pressure A descent", "Max RA Pressure Atrial contraction",
    "Max RA Pressure Tricuspid Opening", "Min RA Pressure V descent",
    "Min LA Volume", "Max LA Volume", "Min LA Pressure A descent", "Max LA Pressure Atrial contraction",
    "Max LA Pressure Mitral Opening", "Min LA Pressure V descent",
    "LA Contraction Volume diff", "RA Contraction Volume diff", "LV Pressure Deriv", "RV Pressure Deriv", "Tidal Volume", "Minute Ventilation",
    "Cardiac Output", "PaO2", "PaCO2", "Percentage Volume Change"]

# X_all = np.load(f'Calibration_31_03_26/test_param_rest_20_26_03.npy', allow_pickle=True)
# Implaus = np.load(f'Calibration_31_03_26/NROY_Implaus_rest_20_26_03.npy', allow_pickle=True)
# keep rows where every implausibility value is < 1
# mask = np.all(Implaus < 1, axis=1)
# X_calibrated = X_all[mask]
# implaus_kept = Implaus[mask]

KDE_file = "Calibration_31_03_26/KDE_prior_posterior.png"

X_calibrated = np.load("Calibration_31_03_26/X_Calibrated.npy")
Result_calibrated = np.load("Calibration_31_03_26/Result_Calibrated.npy")

X_initial = np.load("Calibration_31_03_26/LHCS_1000_X_20.npy")
Result_initial = np.load("Calibration_31_03_26/LHCS_1000_Result_20.npy")


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

    vals_initial = Result_initial[:, i]
    vals_calibrated = Result_calibrated[:, i]

    vals_initial = vals_initial[np.isfinite(vals_initial)]
    vals_calibrated = vals_calibrated[np.isfinite(vals_calibrated)]

    sns.kdeplot(
        vals_initial,
        ax=ax,
        label="Initial",
        fill=True,
        alpha=0.3,
        linewidth=1.5
    )

    sns.kdeplot(
        vals_calibrated,
        ax=ax,
        label="Calibrated",
        fill=True,
        alpha=0.3,
        linewidth=1.5
    )

    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

for j in range(n_outputs, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(KDE_file, dpi=300, bbox_inches="tight")
plt.close()