"""
Plot KDE of prior (NROY) vs posterior predictive distribution per output.

Prior side:     NROY parameter samples pushed through the GP emulators.
Posterior side: pred_check_matrix.npy saved by KNN_MCMC_Rest.py.

The atrial observables at indices 17/18 are stored in pred_check_matrix.npy as
active-emptying fractions, so they are converted back to pre-atrial
contraction volume for display to match the raw emulator outputs.
"""
import math
import os
import numpy as np
import torch
import gpytorch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde


# ================================================================
# SETTINGS -- match your run
# ================================================================
DATE_SUFFIX  = "12_4"
root_folder = "MCMC_HPC"
PERCENT      = 20
EMULATOR_DIR = "Emulator_wave_3"
OUT_DIR      = f"{root_folder}/MCMC_Rest_20_05_05_1500_logspline_copula_prior"          # folder with pred_check_matrix.npy
N_PRIOR      = 2000                          # NROY samples to push through
RANDOM_SEED  = 0
FORCE_REEVALUATE_EMULATORS = False


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFINED_MAP_SAMPLE_PATH = os.path.join(
    SCRIPT_DIR,
    OUT_DIR,
    "refined_map_sample.npy",
)
PRIOR_PREDICTIVE_CACHE = os.path.join(
    OUT_DIR,
    f"prior_predictive_cache_{EMULATOR_DIR}_{PERCENT}_{DATE_SUFFIX}.npz",
)
REMOVED_OUTPUT_INDICES = (11, 14, 17, 20, 27, 30)
PLOT_PERCENTILE_COVERAGE = 99.5
PLOT_COLORS = ("#BBA3D6", "#9DB8D8", "#7DB6C0")
TARGET_LINE_COLOR = "#4A4A4A"
TARGET_BAND_COLOR = "#4A4A4A"
MAP_LINE_COLOR = "#D68484"
SIMULATOR_MAP_LINE_COLOR = "#8A6F4E"
SIMULATOR_HEART_RATE_DIVISOR = 10.0

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "axes.linewidth": 1.1,
})


def _propagated_vpre_display_stats(vmin_mean, vmin_var, vmax_mean, vmax_var,
                                   f_mean, f_var):
    """Display-only mean/std for V_pre = V_min + f * (V_max - V_min).

    This mirrors the zero-cross-covariance assumption used by the MCMC
    observation model: V_min, V_max and the active-emptying fraction f are
    treated as independent when constructing the display normalisation.
    """
    delta_mean = float(vmax_mean) - float(vmin_mean)
    f_mean = float(f_mean)
    f_var = float(f_var)

    vpre_mean = float(vmin_mean) + f_mean * delta_mean
    ef2 = f_mean ** 2 + f_var
    e1mf2 = (1.0 - f_mean) ** 2 + f_var
    vpre_var = (
        e1mf2 * float(vmin_var)
        + ef2 * float(vmax_var)
        + f_var * delta_mean ** 2
    )
    return vpre_mean, math.sqrt(max(vpre_var, 0.0))


def _load_lhcs_results():
    """Load the original LHC training outputs and align them to output_names."""
    lhcs_path = os.path.join(SCRIPT_DIR, "LHCS_Result_20.npy")
    lhcs_result = np.load(lhcs_path)

    if lhcs_result.shape[1] == len(output_names) + len(REMOVED_OUTPUT_INDICES):
        lhcs_result = np.delete(lhcs_result, REMOVED_OUTPUT_INDICES, axis=1)
    elif lhcs_result.shape[1] != len(output_names):
        raise ValueError(
            f"Unexpected LHCS result shape {lhcs_result.shape}; expected "
            f"{len(output_names) + len(REMOVED_OUTPUT_INDICES)} or {len(output_names)} columns."
        )

    return lhcs_result


def _prepare_simulator_map_prediction(raw_output):
    """Align a raw simulator MAP output with the 25 plotted emulator outputs."""
    raw_output = np.asarray(raw_output, dtype=np.float64).ravel()
    expected_full = len(output_names) + len(REMOVED_OUTPUT_INDICES)

    if raw_output.size == len(output_names):
        sim_pred = raw_output.copy()
    elif raw_output.size >= expected_full:
        if raw_output.size > expected_full:
            print(
                f"Simulator MAP output has {raw_output.size} values; using the "
                f"first {expected_full} raw summary outputs before column removal."
            )
        sim_pred = np.delete(raw_output[:expected_full], REMOVED_OUTPUT_INDICES)
    else:
        raise ValueError(
            f"Simulator MAP output has {raw_output.size} values; expected either "
            f"{len(output_names)} plotted outputs or at least {expected_full} raw outputs."
        )

    if sim_pred[0] > 5.0:
        sim_pred[0] /= SIMULATOR_HEART_RATE_DIVISOR

    return sim_pred


def _load_prior_predictive_cache(cache_path, keep, param_idx, theta_keep_np):
    """Load cached emulator predictions when they match this run."""
    if FORCE_REEVALUATE_EMULATORS or not os.path.exists(cache_path):
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            prior_pred = cache["prior_pred"]
            cached_keep = cache["keep"]
            cached_param_idx = cache["param_idx"]
            cached_output_names = cache["output_names"].astype(str).tolist()
            cached_theta_keep = cache["theta_keep"] if "theta_keep" in cache.files else None

        expected_shape = (len(keep), len(output_names))
        if prior_pred.shape != expected_shape:
            print(
                f"Cached prior predictive has shape {prior_pred.shape}, "
                f"expected {expected_shape}; recomputing."
            )
            return None
        if not np.array_equal(cached_keep, keep):
            print("Cached prior predictive uses different kept samples; recomputing.")
            return None
        if not np.array_equal(cached_param_idx, np.asarray(param_idx, dtype=np.int64)):
            print("Cached prior predictive uses different parameter columns; recomputing.")
            return None
        if cached_output_names != output_names:
            print("Cached prior predictive uses different output names; recomputing.")
            return None
        if cached_theta_keep is None or not np.allclose(cached_theta_keep, theta_keep_np):
            print("Cached prior predictive uses different parameter values; recomputing.")
            return None

        print(f"Loaded cached prior predictive from {cache_path}")
        return prior_pred
    except Exception as exc:
        print(f"Could not load prior predictive cache ({exc}); recomputing.")
        return None


def _save_prior_predictive_cache(cache_path, prior_pred, keep, param_idx, theta_keep_np):
    """Save emulator predictions so later plot runs can skip emulator evaluation."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        prior_pred=prior_pred.astype(np.float32, copy=False),
        keep=np.asarray(keep, dtype=np.int64),
        param_idx=np.asarray(param_idx, dtype=np.int64),
        theta_keep=np.asarray(theta_keep_np, dtype=np.float32),
        output_names=np.asarray(output_names, dtype="<U64"),
        emulator_dir=np.asarray(EMULATOR_DIR),
        percent=np.asarray(PERCENT),
        date_suffix=np.asarray(DATE_SUFFIX),
        random_seed=np.asarray(RANDOM_SEED),
    )
    print(f"Saved prior predictive cache to {cache_path}")


def _trim_to_central_percentile(*arrays):
    """Trim all series to a shared central percentile window for one panel."""
    finite_arrays = [arr[np.isfinite(arr)] for arr in arrays]
    all_vals = np.concatenate(finite_arrays)

    tail = 0.5 * (100.0 - PLOT_PERCENTILE_COVERAGE)
    lo, hi = np.percentile(all_vals, [tail, 100.0 - tail])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = all_vals.min(), all_vals.max()

    trimmed = []
    for arr in finite_arrays:
        mask = (arr >= lo) & (arr <= hi)
        arr_trim = arr[mask]
        trimmed.append(arr_trim if arr_trim.size else arr)

    return trimmed, lo, hi


def _draw_distribution(ax, values, xx, color, label, alpha):
    """Draw a filled KDE curve with a histogram fallback."""
    try:
        density = gaussian_kde(values)(xx)
        ax.fill_between(xx, 0.0, density, color=color, alpha=alpha, linewidth=0.0, zorder=3)
        ax.plot(xx, density, color=color, linewidth=2.4, label=label, zorder=4)
    except Exception:
        ax.hist(
            values,
            bins=40,
            density=True,
            alpha=alpha,
            color=color,
            edgecolor=color,
            linewidth=1.0,
            label=label,
            zorder=3,
        )


def _load_emulators():
    """Load the Wave 3 GP emulators used by this plot."""
    emulators = {}
    for name in output_names:
        path = os.path.join(
            f"{root_folder}/",
            EMULATOR_DIR,
            name,
            f"GaussianProcessMatern32_{name}_best.joblib",
        )
        te = joblib.load(path)
        te.model.eval()
        te.model.likelihood.eval()
        emulators[name] = te
    print(f"  Loaded {len(emulators)} emulators")
    return emulators


def _predict_emulator_means(theta_np, emulators):
    """Evaluate emulator means in physical output units for one or more theta rows."""
    theta_np = np.asarray(theta_np, dtype=np.float32)
    single_input = theta_np.ndim == 1
    if single_input:
        theta_np = theta_np[None, :]

    theta = torch.tensor(theta_np, dtype=torch.float32)
    mu = np.zeros((theta.shape[0], len(output_names)), dtype=np.float32)

    with torch.no_grad():
        for i, name in enumerate(output_names):
            te = emulators[name]
            gp = te.model

            x_mean = te.x_transforms[0].mean.detach().squeeze(0)
            x_std = te.x_transforms[0].std.detach().squeeze(0)
            x_t = (theta - x_mean) / x_std

            with gpytorch.settings.fast_pred_var():
                out = gp(x_t)
            mean_t = out.mean

            te_mean = te.y_transforms[0].mean.detach().squeeze()
            te_std = te.y_transforms[0].std.detach().squeeze()
            if gp.y_transform is not None and getattr(gp.y_transform, "_is_fitted", False):
                gp_mean = gp.y_transform.mean.detach().squeeze()
                gp_std = gp.y_transform.std.detach().squeeze()
                y_std_c = gp_std * te_std
                y_mean_c = gp_mean * te_std + te_mean
            else:
                y_std_c = te_std
                y_mean_c = te_mean

            mu[:, i] = (mean_t * y_std_c + y_mean_c).cpu().numpy().squeeze()

    return mu[0] if single_input else mu


def _load_refined_map_sample(sample_path, expected_subset_vars):
    """Load the refined MAP theta and verify it matches the emulator input order."""
    theta = np.load(sample_path)
    expected_subset_vars = list(expected_subset_vars)

    if theta.shape != (len(expected_subset_vars),):
        raise ValueError(
            f"Refined MAP sample has shape {theta.shape}; expected "
            f"({len(expected_subset_vars)},) for the current subset."
        )

    subset_vars_path = os.path.join(os.path.dirname(sample_path), "subset_vars.npy")
    if os.path.exists(subset_vars_path):
        map_subset_vars = np.load(subset_vars_path, allow_pickle=True).tolist()
        if map_subset_vars != expected_subset_vars:
            raise ValueError(
                "Parameter order mismatch between refined MAP subset_vars.npy "
                "and the KDE emulator inputs."
            )

    return theta


LA_MIN_IDX, LA_MAX_IDX, LA_PRE_IDX = 13, 14, 17
RA_MIN_IDX, RA_MAX_IDX, RA_PRE_IDX = 9, 10, 18


output_names = [
    "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV",
    "ESV", "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure",
    "Min_RV_Pressure", "Min_RA_Volume", "Max_RA_Volume",
    "Max_RA_Pressure_Atrial_contraction",
    "Max_RA_Pressure_Tricuspid_Opening", "Min_LA_Volume",
    "Max_LA_Volume", "Max_LA_Pressure_Atrial_contraction",
    "Max_LA_Pressure_Mitral_Opening", "LA_Contraction_Volume_diff",
    "RA_Contraction_Volume_diff", "LV_Pressure_Deriv",
    "RV_Pressure_Deriv", "Tidal_Volume", "Minute_Ventilation",
    "PaO2", "PaCO2",
]

display_specs = {
    "Heart_Rate": ("Heart Rate", "BPM", 60.0),
    "Systolic_Pressure": ("LV Systolic Pressure", "mmHg", 1.0),
    "Diastolic_Pressure": ("LV Diastolic Pressure", "mmHg", 1.0),
    "EDV": ("LV End-Diastolic Volume", "mL", 1.0),
    "ESV": ("LV End-Systolic Volume", "mL", 1.0),
    "Max_RV_Volume": ("RV End-Diastolic Volume", "mL", 1.0),
    "Min_RV_Volume": ("RV End-Systolic Volume", "mL", 1.0),
    "Max_RV_Pressure": ("RV Systolic Pressure", "mmHg", 1.0),
    "Min_RV_Pressure": ("RV Diastolic Pressure", "mmHg", 1.0),
    "Min_RA_Volume": ("Min RA Volume", "mL", 1.0),
    "Max_RA_Volume": ("Max RA Volume", "mL", 1.0),
    "Max_RA_Pressure_Atrial_contraction": ("Max RA Pressure A wave", "mmHg", 1.0),
    "Max_RA_Pressure_Tricuspid_Opening": ("Max RA Pressure V wave", "mmHg", 1.0),
    "Min_LA_Volume": ("Min LA Volume", "mL", 1.0),
    "Max_LA_Volume": ("Max LA Volume", "mL", 1.0),
    "Max_LA_Pressure_Atrial_contraction": ("Max LA Pressure A wave", "mmHg", 1.0),
    "Max_LA_Pressure_Mitral_Opening": ("Max LA Pressure V wave", "mmHg", 1.0),
    "LA_Contraction_Volume_diff": ("LA Pre-Atrial Contraction Volume", "mL", 1.0),
    "RA_Contraction_Volume_diff": ("RA Pre-Atrial Contraction Volume", "mL", 1.0),
    "LV_Pressure_Deriv": ("Max LV Pressure Derivative", "mmHg/s", 1.0),
    "RV_Pressure_Deriv": ("Max RV Pressure Derivative", "mmHg/s", 1.0),
    "Tidal_Volume": ("Tidal Volume", "L", 1.0),
    "Minute_Ventilation": ("Minute Ventilation", "L/min", 1.0),
    "PaO2": (r"PaO$_2$", "mmHg", 1.0),
    "PaCO2": (r"PaCO$_2$", "mmHg", 1.0),
}

observation = {
    "Heart_Rate": (1.23, 0.05),
    "Systolic_Pressure": (123, 324),
    "Diastolic_Pressure": (76.7, 65.61),
    "EDV": (152.1, 767.29),
    "ESV": (62.3, 243.36),
    "Max_RV_Volume": (151.9, 1004.89),
    "Min_RV_Volume": (64.4, 299.29),
    "Max_RV_Pressure": (22.5, 56.25),
    "Min_RV_Pressure": (4.0, 9.0),
    "Min_RA_Volume": (45.7, 125.44),
    "Max_RA_Volume": (92.4, 380.25),
    "Max_RA_Pressure_Atrial_contraction": (8.0, 9.0),
    "Max_RA_Pressure_Tricuspid_Opening": (5.0, 9.0),
    "Min_LA_Volume": (30.6, 84.64),
    "Max_LA_Volume": (68.3, 306.25),
    "Max_LA_Pressure_Atrial_contraction": (13.0, 9.0),
    "Max_LA_Pressure_Mitral_Opening": (12.0, 9.0),
    "LA_Contraction_Volume_diff": (0.25, 0.0025),
    "RA_Contraction_Volume_diff": (0.25, 0.0025),
    "LV_Pressure_Deriv": (1461.0, 146689.0),
    "RV_Pressure_Deriv": (271.0, 3025.0),
    "Tidal_Volume": (0.850, 0.16),
    "Minute_Ventilation": (11.4, 15.21),
    "PaO2": (102.3, 125.44),
    "PaCO2": (35.5, 24.01),
}

LA_PRE_DISPLAY_MEAN, LA_PRE_DISPLAY_STD = _propagated_vpre_display_stats(
    observation["Min_LA_Volume"][0],
    observation["Min_LA_Volume"][1],
    observation["Max_LA_Volume"][0],
    observation["Max_LA_Volume"][1],
    observation["LA_Contraction_Volume_diff"][0],
    observation["LA_Contraction_Volume_diff"][1],
)
RA_PRE_DISPLAY_MEAN, RA_PRE_DISPLAY_STD = _propagated_vpre_display_stats(
    observation["Min_RA_Volume"][0],
    observation["Min_RA_Volume"][1],
    observation["Max_RA_Volume"][0],
    observation["Max_RA_Volume"][1],
    observation["RA_Contraction_Volume_diff"][0],
    observation["RA_Contraction_Volume_diff"][1],
)

subset_vars_set = {
    'a2', 'ahead1', 'beta2', 'C2', 'C_jp', 'C_O2_param1', 'C_sv',
    'Cvam_O2_n', 'E_rs', 'Emax_la', 'Emax_lv0', 'Emax_ra', 'Emax_rv0',
    'f_ab_max', 'fab_o', 'fall_time_ven', 'fes_inf', 'fes_min', 'fes_o',
    'fev_inf', 'fev_o', 'GT_s', 'GT_v', 'Io_met', 'Io_sv', 'K2',
    'k_ab', 'kcc_sv', 'KE_la', 'KE_lv', 'KE_ra', 'KE_rv', 'kes',
    'kmet', 'Kv_mi', 'Kv_po', 'Kv_tr', 'l', 'MO2_bp', 'P0_la',
    'P0_lv', 'P0_ra', 'P0_rv', 'P_n', 'PaCO2_n', 'r', 'R_pa', 'R_pp',
    'R_rs', 'R_sa', 'rise_time_atr', 'rise_time_ven', 'Rvc_n', 'T0',
    'theta_svn', 'V0_dead', 'V_nominal', 'V_scale', 'Vu_amv0', 'Vu_bv',
    'Vu_ev0', 'Vu_jp', 'Vu_la', 'Vu_lv', 'Vu_ra', 'Vu_rv', 'Vu_sv0',
    'Wb_sh', 'Wb_sv',
}

# # total blood volume
# subset_vars_set = {
#     'a2', 'ahead1', 'beta2', 'C2', 'C_O2_param1', 'Cvam_O2_n', 'E_rs', 'Emax_la', 'Emax_lv0', 'Emax_rv0',
#                'f_ab_max', 'fall_time_ven', 'fes_inf', 'fes_min', 'fes_o', 'fev_o', 'GT_s', 'GT_v', 'Io_met', 'K2',
#                'k_ab', 'KE_la', 'KE_lv', 'KE_ra', 'KE_rv', 'kes', 'l', 'MO2_bp', 'P0_la', 'P0_lv', 'P0_rv', 'P_n',
#                'PaCO2_n', 'r', 'R_rs', 'R_sa', 'rise_time_ven', 'T0', 'V0_dead', 'V_nominal', 'V_scale', 'V_tot',
#                'Vu_ev0', 'Vu_jp', 'Vu_la', 'Vu_lv', 'Vu_ra', 'Vu_rv', 'Vu_sv0', 'Wb_sh'
# }



# ================================================================
# LOAD TESTED PARAMS, IMPLAUSIBILITY SCORES, EMULATORS
# ================================================================
print("Loading tested params + saved implausibility scores...")
nroy_params_dict = np.load(
    f"{root_folder}/NROY_Params_rest_{PERCENT}_{DATE_SUFFIX}.npy", allow_pickle=True
).item()
all_param_names = list(nroy_params_dict.keys())
subset_vars     = [n for n in all_param_names if n in subset_vars_set]
param_idx       = [all_param_names.index(n) for n in subset_vars]

test_param = np.load(f"{root_folder}/test_param_rest_{PERCENT}_{DATE_SUFFIX}.npy")   # (N_all, 272)
impl       = np.load(f"{root_folder}/NROY_Implaus_rest_{PERCENT}_{DATE_SUFFIX}.npy") # (N_all, 25)
assert test_param.shape[0] == impl.shape[0], (
    f"Row mismatch: test_param {test_param.shape} vs implaus {impl.shape}"
)
print(f"  Tested points: {test_param.shape[0]}")


# ================================================================
# PICK LOWEST-IMPLAUSIBILITY POINTS (rank-1 / worst-case per sample)
# Only these are pushed through the emulators for predictive sampling.
# ================================================================
worst = impl.max(axis=1)                        # (N_all,)
# N = min(N_PRIOR, len(worst))
# keep1 = np.argsort(worst)
# worst1 = worst[keep1]
# keep = np.argsort(worst)[:N]
# worst2 = worst[keep]
keep1 = np.where(worst < 3)[0]
keep = keep1[np.argsort(worst[keep1])]
N_PRIOR      = len(keep)                        # NROY samples to push through
N = N_PRIOR
print(f"  Kept {N} lowest-implausibility points "
      f"(worst-case I range: {worst[keep].min():.3f} - {worst[keep].max():.3f})")

theta_keep_np = test_param[np.ix_(keep, param_idx)]


# ================================================================
# PRIOR PREDICTIVE -- push ONLY the kept points through the emulators
# ================================================================
n_out = len(output_names)
prior_pred = _load_prior_predictive_cache(
    PRIOR_PREDICTIVE_CACHE, keep, param_idx, theta_keep_np
)
emulators = None

if prior_pred is None:
    emulators = _load_emulators()
    print(f"Pushing {N} kept samples through emulators...")
    mu_sel = _predict_emulator_means(theta_keep_np, emulators)

    prior_pred = mu_sel
    _save_prior_predictive_cache(
        PRIOR_PREDICTIVE_CACHE, prior_pred, keep, param_idx, theta_keep_np
    )


# ================================================================
# POSTERIOR PREDICTIVE (already saved)
# ================================================================
post_pred = np.load(os.path.join(OUT_DIR, "pred_check_matrix.npy"))
lhcs_pred = _load_lhcs_results()
print(f"Posterior predictive: {post_pred.shape}")
print(f"Initial LHCS results: {lhcs_pred.shape}")

plot_lhcs_pred = lhcs_pred.copy()
plot_prior_pred = prior_pred.copy()
plot_post_pred = post_pred.copy()
plot_target_means = np.array([observation[name][0] for name in output_names], dtype=np.float64)
plot_target_stds = np.sqrt(
    np.array([observation[name][1] for name in output_names], dtype=np.float64)
)

# KNN_MCMC_Rest.py saves the atrial columns as active-emptying fractions.
# Convert them back to the display-space pre-atrial-contraction volumes so
# the posterior curves line up with the raw emulator outputs used on the prior
# side of this figure.
plot_post_pred[:, LA_PRE_IDX] = (
    post_pred[:, LA_MIN_IDX]
    + post_pred[:, LA_PRE_IDX] * (post_pred[:, LA_MAX_IDX] - post_pred[:, LA_MIN_IDX])
)
plot_post_pred[:, RA_PRE_IDX] = (
    post_pred[:, RA_MIN_IDX]
    + post_pred[:, RA_PRE_IDX] * (post_pred[:, RA_MAX_IDX] - post_pred[:, RA_MIN_IDX])
)
plot_target_means[LA_PRE_IDX] = LA_PRE_DISPLAY_MEAN
plot_target_means[RA_PRE_IDX] = RA_PRE_DISPLAY_MEAN
plot_target_stds[LA_PRE_IDX] = LA_PRE_DISPLAY_STD
plot_target_stds[RA_PRE_IDX] = RA_PRE_DISPLAY_STD

refined_map_theta = _load_refined_map_sample(REFINED_MAP_SAMPLE_PATH, subset_vars)
if emulators is None:
    print("Loading emulators for refined MAP overlay...")
    emulators = _load_emulators()
plot_refined_map_pred = _predict_emulator_means(refined_map_theta, emulators)
print(f"Refined MAP Wave 3 emulator prediction: {plot_refined_map_pred.shape}")


# ================================================================
# Evaluate MAP using the Simulator to demonstrate emulator accuracy
# ================================================================
Simulator_Output_for_Corresponding_MAP = np.array([
    1.073305682542691, 113.17398369206448, 77.89237364931813, 133.06283917814605, 53.04661911524478, 138.73297055885527,
    58.69822637535621, 32.05952474134211, 2.5610435254776327, 38.58561046611903, 87.31461310805685, 5.165071233079443,
    10.724934963518619, 5.451653374057925, 4.417982662948159, 32.65103372634101, 79.89466118889904, 9.933809285882843,
    12.497628083327339, 11.332233210168654, 8.363591107808544, 43.241154006862, 50.1624056374744, 1148.4677866551742,
    270.1560615868634, 0.7184874643157888, 9.187652860845436, 93.07063496030592, 104.43613721988764, 34.3427606426357,
    0.27377222713552885
])
plot_simulator_map_pred = _prepare_simulator_map_prediction(
    Simulator_Output_for_Corresponding_MAP
)
print(f"Simulator MAP prediction aligned to plotted outputs: {plot_simulator_map_pred.shape}")


# ================================================================
# PLOT
# ================================================================
n = len(output_names)
ncol = 5
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.2 * nrow))
axes = axes.ravel()

for i, name in enumerate(output_names):
    ax = axes[i]
    label_name, unit, scale = display_specs[name]
    lh = plot_lhcs_pred[:, i] * scale
    pr = plot_prior_pred[:, i] * scale
    po = plot_post_pred[:, i] * scale
    m = plot_target_means[i] * scale
    s = plot_target_stds[i] * scale
    map_m = plot_refined_map_pred[i] * scale
    sim_map_m = plot_simulator_map_pred[i] * scale

    (lh, pr, po), lo, hi = _trim_to_central_percentile(lh, pr, po)
    xx = np.linspace(lo, hi, 500)

    ax.axvspan(
        m - s, m + s,
        facecolor=TARGET_BAND_COLOR,
        alpha=0.05,
        linewidth=0.0,
        zorder=0,
    )
    # ax.axvline(m - s, color=TARGET_BAND_COLOR, linestyle=(0, (5, 3)),
    #            linewidth=1.4, alpha=0.75, zorder=1)
    # ax.axvline(m + s, color=TARGET_BAND_COLOR, linestyle=(0, (5, 3)),
    #            linewidth=1.4, alpha=0.75, zorder=1)
    ax.axvline(m, color=TARGET_LINE_COLOR, linestyle="-",
               linewidth=1.8, alpha=0.75, zorder=2)
    ax.axvline(map_m, color=MAP_LINE_COLOR, linestyle=(0, (2, 2)),
               linewidth=2.5, alpha=0.9, zorder=5)
    ax.axvline(sim_map_m, color=SIMULATOR_MAP_LINE_COLOR, linestyle=(0, (2, 2)),
               linewidth=2.4, alpha=0.95, zorder=6)

    _draw_distribution(ax, lh, xx, PLOT_COLORS[0], "Initial LHCS", 0.38)
    _draw_distribution(ax, pr, xx, PLOT_COLORS[1], "NROY prior", 0.38)
    _draw_distribution(ax, po, xx, PLOT_COLORS[2], "Posterior", 0.42)

    ax.set_xlabel(f"{label_name}\n({unit})", fontsize=20, labelpad=10)
    ax.set_ylabel("Density" if i % ncol == 0 else "", fontsize=20)
    ax.set_title("")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", labelsize=18, width=1.0, length=4, colors="#303030")
    ax.tick_params(axis="y", labelleft=(i % ncol == 0))
    ax.set_yticks([])
    # ax.grid(axis="y", color="#D9D9D9", linewidth=0.9, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.margins(y=0.08)

for k in range(n, len(axes)):
    axes[k].axis("off")

legend_handles = [
    Patch(facecolor=PLOT_COLORS[0], edgecolor=PLOT_COLORS[0], alpha=0.38, label="Initial LHCS"),
    Patch(facecolor=PLOT_COLORS[1], edgecolor=PLOT_COLORS[1], alpha=0.38, label="NROY prior"),
    Patch(facecolor=PLOT_COLORS[2], edgecolor=PLOT_COLORS[2], alpha=0.42, label="Posterior"),
    Line2D([0], [0], color=TARGET_LINE_COLOR, linewidth=2.2, label="Target mean"),
    Patch(facecolor=TARGET_BAND_COLOR, edgecolor=TARGET_BAND_COLOR, alpha=0.10, label=r"Target $\pm$ 1 SD"),
    Line2D([0], [0], color=MAP_LINE_COLOR, linestyle=(0, (2, 2)),
           linewidth=2.4, label="MAP (Wave 3 Emulator)"),
    Line2D([0], [0], color=SIMULATOR_MAP_LINE_COLOR, linestyle=(0, (2, 2)),
           linewidth=2.4, label="MAP (simulator)"),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.975),
    ncol=7,
    frameon=False,
    fontsize=20,
    handlelength=1.8,
    columnspacing=1.4,
)
plt.tight_layout(rect=(0.02, 0.02, 1.0, 0.94))
out_path = os.path.join(OUT_DIR, "KDE_prior_posterior.png")
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Saved {out_path}")
