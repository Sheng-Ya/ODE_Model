"""
Plot KDE of prior (NROY) vs posterior predictive distribution per output.

Prior side:     NROY parameter samples pushed through the GP emulators.
Posterior side: pred_check_matrix.npy saved by KNN_MCMC_Rest.py.

Both use the *same* emulators, so the comparison is apples-to-apples.
"""
import os
import numpy as np
import torch
import gpytorch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ================================================================
# SETTINGS -- match your run
# ================================================================
DATE_SUFFIX  = "12_4"
root_folder = "three_implaus_pre_A_calib"
PERCENT      = 20
EMULATOR_DIR = "Emulator_wave_1wave"
OUT_DIR      = f"{root_folder}/MCMC_Rest_20_21_04_3000_logspline_copula_prior"          # folder with pred_check_matrix.npy
N_PRIOR      = 2000                          # NROY samples to push through
RANDOM_SEED  = 0


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
    "Min_RA_Volume": (30.6, 76.4),
    "Max_RA_Volume": (92.4, 380.25),
    "Max_RA_Pressure_Atrial_contraction": (8.0, 9.0),
    "Max_RA_Pressure_Tricuspid_Opening": (5.0, 9.0),
    "Min_LA_Volume": (32.9, 75.69),
    "Max_LA_Volume": (68.3, 306.25),
    "Max_LA_Pressure_Atrial_contraction": (13.0, 9.0),
    "Max_LA_Pressure_Mitral_Opening": (12.0, 9.0),
    "LA_Contraction_Volume_diff": (41.8, 62.41),
    "RA_Contraction_Volume_diff": (46.1, 73.96),
    "LV_Pressure_Deriv": (1461.0, 146689.0),
    "RV_Pressure_Deriv": (271.0, 3025.0),
    "Tidal_Volume": (0.850, 0.16),
    "Minute_Ventilation": (11.4, 15.21),
    "PaO2": (102.3, 125.44),
    "PaCO2": (35.5, 24.01),
}

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
print("Loading tested params + saved implausibility scores + emulators...")
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

emulators = {}
for name in output_names:
    path = os.path.join(f"{root_folder}/", EMULATOR_DIR, name,
                        f"GaussianProcessMatern32_{name}_best.joblib")
    emulators[name] = joblib.load(path)
print(f"  Tested points: {test_param.shape[0]}  Emulators: {len(emulators)}")


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

theta_keep = torch.tensor(test_param[np.ix_(keep, param_idx)],
                          dtype=torch.float32)


# ================================================================
# PRIOR PREDICTIVE -- push ONLY the kept points through the emulators
# ================================================================
n_out = len(output_names)
mu_sel  = np.zeros((N, n_out), dtype=np.float32)
var_sel = np.zeros((N, n_out), dtype=np.float32)

print(f"Pushing {N} kept samples through emulators...")
with torch.no_grad():
    for i, name in enumerate(output_names):
        te = emulators[name]
        gp = te.model
        gp.eval(); gp.likelihood.eval()

        x_mean = te.x_transforms[0].mean.detach().squeeze(0)
        x_std  = te.x_transforms[0].std.detach().squeeze(0)
        x_t = (theta_keep - x_mean) / x_std

        with gpytorch.settings.fast_pred_var():
            out = gp(x_t)
        mean_t = out.mean
        var_t  = out.variance.clamp(min=1e-10)

        te_mean = te.y_transforms[0].mean.detach().squeeze()
        te_std  = te.y_transforms[0].std.detach().squeeze()
        if gp.y_transform is not None and getattr(gp.y_transform, "_is_fitted", False):
            gp_mean = gp.y_transform.mean.detach().squeeze()
            gp_std  = gp.y_transform.std.detach().squeeze()
            y_std_c = gp_std * te_std
            y_mean_c = gp_mean * te_std + te_mean
        else:
            y_std_c = te_std
            y_mean_c = te_mean

        mu_sel[:, i]  = (mean_t * y_std_c + y_mean_c).cpu().numpy().squeeze()
        var_sel[:, i] = (var_t * y_std_c ** 2).cpu().numpy().squeeze()

# Draw one predictive sample per kept point (includes emulator uncertainty)
rng = np.random.default_rng(RANDOM_SEED)
prior_pred = mu_sel #+ rng.standard_normal(mu_sel.shape).astype(np.float32) * np.sqrt(var_sel)


# ================================================================
# POSTERIOR PREDICTIVE (already saved)
# ================================================================
post_pred = np.load(os.path.join(OUT_DIR, "pred_check_matrix.npy"))
print(f"Posterior predictive: {post_pred.shape}")


# ================================================================
# PLOT
# ================================================================
n = len(output_names)
ncol = 5
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3*nrow))
axes = axes.ravel()

for i, name in enumerate(output_names):
    ax = axes[i]
    pr, po = prior_pred[:, i], post_pred[:, i]
    m, s = observation[name]
    s = np.sqrt(s)

    all_vals = np.concatenate([pr, po])
    lo, hi = all_vals.min(), all_vals.max()
    # pad = 0.001 * (hi - lo)
    # xx = np.linspace(lo - pad, hi + pad, 400)

    # lo, hi = np.percentile(np.concatenate([pr, po]), [0.5, 99.5])
    xx = np.linspace(lo, hi, 400)

    try:
        ax.fill_between(xx, gaussian_kde(pr)(xx), alpha=0.4,
                        color="C0", label="NROY prior")
    except Exception:
        ax.hist(pr, bins=40, density=True, alpha=0.4, color="C0",
                label="NROY prior")
    try:
        ax.fill_between(xx, gaussian_kde(po)(xx), alpha=0.5,
                        color="C1", label="posterior")
    except Exception:
        ax.hist(po, bins=40, density=True, alpha=0.5, color="C1",
                label="posterior")

    ax.axvline(m, color="k", ls="--", lw=1)
    ax.axvspan(m - s, m + s, color="grey", alpha=0.2)
    ax.set_title(name.replace("_", " "), fontsize=9)
    ax.set_yticks([])
    if i == 0:
        ax.legend(fontsize=7)

for k in range(n, len(axes)):
    axes[k].axis("off")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "KDE_prior_posterior.png")
plt.savefig(out_path, dpi=200)
plt.close()
print(f"Saved {out_path}")
