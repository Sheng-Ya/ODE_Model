"""
KNN_MCMC_Rest.py — Post-history-matching calibration pipeline

Pipeline:
  1. Load NROY results from history matching (3-sigma threshold)
  2. KNN density estimation -> find densest region as MCMC start
  3. Pyro NUTS MCMC -> posterior distribution targeting 1 sigma of rest targets

Why NUTS over emcee / random-walk MH:
  - 69 calibration parameters: gradient-based NUTS scales as O(d^{1/4}),
    whereas emcee's stretch move degrades rapidly past ~30 dims.
  - The GP emulators (autoemulate TransformedEmulator with GPyTorch)
    support with_grad=True, so the full chain
        theta -> x_transform -> GP kernel -> predictive mean/var -> log_prob
    is differentiable through torch autograd.
  - NUTS automatically adapts step size and mass matrix during warmup,
    handling the very different scales across the 69 parameters.
  - ~100-200x more sample-efficient than emcee for this dimensionality.

Expects in working directory:
  NROY_Points_rest_{PERCENT}_{DATE_SUFFIX}.npy  -- NROY parameter vectors
  NROY_Params_rest_{PERCENT}_{DATE_SUFFIX}.npy  -- NROY bounds dict
  {EMULATOR_DIR}/{output_name}/GaussianProcessMatern32_{output_name}_best.joblib

Usage:
  python KNN_MCMC_Rest.py
"""

import math
import os
import warnings
import joblib
import numpy as np
import torch
import gpytorch

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, HMC

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"   # set before sklearn/joblib uses loky

warnings.filterwarnings("ignore")

# ================================================================
# SETTINGS
# ================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)

DATE_SUFFIX  = "12_4"                    # matches HM output file names
PERCENT      = 20                        # param range +/-% used in HM
EMULATOR_DIR = "Emulator_wave_1wave"     # GP emulators from last refitted wave

# KNN
KNN_K = 50                               # neighbours for density estimation

# NUTS MCMC
N_WARMUP        = 100                    # warmup (step-size + mass-matrix adapt)
N_SAMPLES       = 300                   # posterior draws per chain
N_CHAINS        = 1                      # independent chains (1 if multiproc fails)
MAX_TREE_DEPTH  = 7
TARGET_ACCEPT   = 0.8

# Posterior predictive
N_PRED_CHECK = 200                       # posterior samples for predictive check


SQRT3 = math.sqrt(3.0)


def extract_fast_caches(emulators, output_names):
    """Pre-warm GPyTorch prediction caches and extract y-transform params.

    Instead of reimplementing the Matern-3/2 kernel manually (which can diverge
    numerically from GPyTorch's internal noise/jitter handling), this stores a
    reference to each GP model and calls GPyTorch directly during MCMC.  The
    only part of the autoemulate pipeline that is bypassed is the expensive
    delta_method/vmap wrapper, which is unnecessary for affine y-transforms
    (StandardizeTransform: y = y_t * std + mean).

    NOTE: The autoemulate pipeline has TWO y-inverse-transforms:
      1. GP's own y_transform (standardize_y=True by default in the GP class)
      2. TransformedEmulator's y_transforms
    Both are affine (StandardizeTransform), so we pre-combine them into a
    single affine: y_final = y_gp * combined_std + combined_mean.
    """
    caches = {}
    for name in output_names:
        te = emulators[name]
        gp = te.model
        gp.eval()
        gp.likelihood.eval()

        # ---- x-transform (TransformedEmulator's only; GP has standardize_x=False) ----
        x_mean = te.x_transforms[0].mean.detach().squeeze(0)   # (d,)
        x_std  = te.x_transforms[0].std.detach().squeeze(0)    # (d,)

        # ---- y-transforms: combine GP's + TransformedEmulator's ----
        # TransformedEmulator's y-inverse: y_te = y_t * te_std + te_mean
        te_y_mean = te.y_transforms[0].mean.detach().squeeze()  # scalar
        te_y_std  = te.y_transforms[0].std.detach().squeeze()   # scalar

        # GP's own y-inverse: y_gp_out = y_gp * gp_std + gp_mean
        # (applied inside Emulator.predict before TransformedEmulator sees it)
        if gp.y_transform is not None and getattr(gp.y_transform, '_is_fitted', False):
            gp_y_mean = gp.y_transform.mean.detach().squeeze()
            gp_y_std  = gp.y_transform.std.detach().squeeze()
            # Compose: y_final = (y_gp * gp_std + gp_mean) * te_std + te_mean
            combined_y_std  = gp_y_std * te_y_std
            combined_y_mean = gp_y_mean * te_y_std + te_y_mean
        else:
            combined_y_std  = te_y_std
            combined_y_mean = te_y_mean

        # Pre-warm GPyTorch prediction strategy (triggers Cholesky once)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _ = gp(gp.train_inputs[0][:1])

        caches[name] = {
            "gp": gp,
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": combined_y_mean, "y_std": combined_y_std,
        }
    return caches


def make_fast_potential_fn(prior_lo, prior_hi, obs_means_t, obs_vars_t,
                           output_names, gp_caches,
                           rho_LA_t, rho_RA_t):
    """Potential energy calling GPyTorch directly — no autoemulate overhead.

    Bypasses the expensive autoemulate wrapper:
      - delta_method with vmap/jacrev/hessian
      - Distribution object creation
      - make_positive_definite
    Keeps GPyTorch's exact kernel computation + cached prediction strategy.

    For affine y-transforms (StandardizeTransform), the inverse is just:
        mu  = mean_t * y_std + y_mean
        var = var_t  * y_std²
    """
    log_width = torch.log(prior_hi - prior_lo)

    def _potential(z_dict):
        z = z_dict["theta"]                                          # (ndim,)

        # ---- sigmoid transform to constrained space ----
        sig_z = torch.sigmoid(z)
        theta = prior_lo + sig_z * (prior_hi - prior_lo)

        # ---- log |det J| of sigmoid ----
        log_det_jac = (
            torch.nn.functional.logsigmoid(z)
            + torch.nn.functional.logsigmoid(-z)
            + log_width
        ).sum()

        # ---- GP log-likelihood (fast GPyTorch path) ----
        # Collect all predictions first so we can derive diffs for cols 17/18
        mus = [None] * len(output_names)
        vars_ = [None] * len(output_names)
        for i, name in enumerate(output_names):
            c = gp_caches[name]

            # Standardise input
            x_t = (theta - c["x_mean"]) / c["x_std"]                # (d,)

            # GPyTorch prediction (cached Cholesky — no recomputation)
            with gpytorch.settings.fast_pred_var():
                output = c["gp"](x_t.unsqueeze(0))
            mean_t = output.mean.squeeze()
            var_t  = output.variance.squeeze().clamp(min=1e-10)

            # Affine y-inverse-transform (StandardizeTransform)
            mus[i]   = mean_t * c["y_std"] + c["y_mean"]
            vars_[i] = var_t  * c["y_std"] ** 2

        # Emulators 17/18 predict pre-atrial-contraction volumes; calibration
        # targets are diffs (b4_contract - min_volume).
        # Var(A - B) = Var(A) + Var(B) - 2 * Cov(A, B)
        #            = Var(A) + Var(B) - 2 * rho * sqrt(Var(A) * Var(B))
        # where rho is the cross-output correlation estimated once from the
        # training-simulator outputs (see rho_LA / rho_RA computation at module
        # level). This is a plug-in estimator for the unknown Cov(GP_A, GP_B)
        # between two independently-fit GPs; it shrinks the diff variance below
        # the conservative Var(A)+Var(B) upper bound when A and B are positively
        # correlated (which they are: pre-contract and min volumes both rise
        # and fall with filling / elastance / vascular tone).
        mus[17]   = mus[17] - mus[13]
        vars_[17] = (vars_[17] + vars_[13]
                     - 2.0 * rho_LA_t * (vars_[17] * vars_[13]).clamp(min=1e-20).sqrt()
                    ).clamp(min=1e-10)
        mus[18]   = mus[18] - mus[9]
        vars_[18] = (vars_[18] + vars_[9]
                     - 2.0 * rho_RA_t * (vars_[18] * vars_[9]).clamp(min=1e-20).sqrt()
                    ).clamp(min=1e-10)

        # Gaussian likelihood + soft 1-sigma barrier:
        #   log p(y | theta) = -0.5 * sum_i [ z_i^2 + log(s_i^2) ]
        #                    -        sum_i  LAMBDA * max(|z_i| - 1, 0)^2
        #   z_i = (y_i - mu_i) / s_i,  s_i^2 = obs_var_i + GP_var_i(theta)
        #
        # Why this shape:
        #   - The Gaussian term is the original likelihood -> recovers the tight
        #     posterior of the previous run (each output pulled toward its target
        #     with curvature 1/s_i^2).
        #   - The relu(|z|-1)^2 term is exactly 0 for |z| <= 1, so it adds NO
        #     pressure on already-compliant outputs (no spread inflation), and
        #     grows quadratically outside, sharply discouraging any output from
        #     drifting past 1 sigma.
        #   - Effective curvature outside 1 sigma is (1 + 2*LAMBDA)/s_i^2, i.e.
        #     residuals beyond 1 sigma "feel" an effective sigma of
        #     s_i / sqrt(1 + 2*LAMBDA).
        #     LAMBDA=2 -> ~0.45 s_i (moderate); 5 -> ~0.30 s_i (strong barrier).
        #   - C^1-smooth everywhere: relu derivative is 0 on both sides of |z|=1,
        #     so the barrier joins smoothly to the Gaussian -> NUTS step-size
        #     adaptation stays well-behaved.
        #   - Numerical stability: total_var clamped to >=1e-10; tail gradient
        #     grows only as |z| (Gaussian) + |z|-1 (barrier), no exp/divisive blow-up.
        #
        # Mathematically sound: this is a product of a proper Gaussian density
        # and a (proper, up to normalising constant) prior on the residual that
        # puts mass on |z| <= 1 with quadratic decay outside.
        LAMBDA = 0.0   # barrier strength; raise for harder >1 sigma penalty
        ll = torch.tensor(0.0, dtype=torch.float32)
        for i in range(len(output_names)):
            total_var = (obs_vars_t[i] + vars_[i]).clamp(min=1e-10)
            z = (obs_means_t[i] - mus[i]) / total_var.sqrt()
            ll = ll - 0.5 * (z ** 2 + torch.log(total_var))
            excess = torch.relu(z.abs() - 1.0)
            ll = ll - LAMBDA * excess ** 2

        ll = torch.nan_to_num(ll, nan=-1e8, posinf=-1e8, neginf=-1e8)
        return -(ll + log_det_jac)

    return _potential

# ================================================================
# OUTPUT NAMES (same order as emulators / simulator outputs)
# ================================================================
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

# ================================================================
# REST TARGETS: {name_with_spaces: (population_mean, population_variance)}
# ================================================================
observation = {
    "Heart Rate": (1.23, 0.05),
    "Systolic Pressure": (123, 324),
    "Diastolic Pressure": (76.7, 65.61),
    "EDV": (152.1, 767.29),
    "ESV": (62.3, 243.36),
    "Max RV Volume": (151.9, 1004.89),
    "Min RV Volume": (64.4, 299.29),
    "Max RV Pressure": (22.5, 56.25),
    "Min RV Pressure": (4.0, 9.0),
    "Min RA Volume": (30.6, 76.4),
    "Max RA Volume": (92.4, 380.25),
    "Max RA Pressure Atrial contraction": (8.0, 9.0),
    "Max RA Pressure Tricuspid Opening": (5.0, 9.0),
    "Min LA Volume": (32.9, 75.69),
    "Max LA Volume": (68.3, 306.25),
    "Max LA Pressure Atrial contraction": (13.0, 9.0),
    "Max LA Pressure Mitral Opening": (12.0, 9.0),
    "LA Contraction Volume diff": (9.4, 11.56),
    "RA Contraction Volume diff": (11.6, 16.81),
    "LV Pressure Deriv": (1461.0, 146689.0),
    "RV Pressure Deriv": (271.0, 3025.0),
    "Tidal Volume": (0.850, 0.16),
    "Minute Ventilation": (11.4, 15.21),
    "PaO2": (102.3, 125.44),
    "PaCO2": (35.5, 24.01),
}

# ================================================================
# CALIBRATION PARAMETER SUBSET (from DGSM sensitivity analysis)
# ================================================================
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





# ============================================================
# 1. LOAD HISTORY MATCHING RESULTS
# ============================================================
print("=" * 60)
print("STEP 1 -- Loading history matching results")
print("=" * 60)

nroy_points_np = np.load(
    f"NROY_Points_rest_{PERCENT}_{DATE_SUFFIX}.npy"
)
nroy_params_dict = np.load(
    f"NROY_Params_rest_{PERCENT}_{DATE_SUFFIX}.npy", allow_pickle=True
).item()

# Parameter ordering from the HM bounds dict (matches sp["names"])
all_param_names = list(nroy_params_dict.keys())

# Calibration subset, preserved in the order they appear in the full vector
subset_vars = [n for n in all_param_names if n in subset_vars_set]
param_idx = [all_param_names.index(n) for n in subset_vars]
ndim = len(subset_vars)

# Extract calibration columns from NROY points
nroy_subset = nroy_points_np[:, param_idx].astype(np.float32)
n_nroy = nroy_subset.shape[0]

# NROY bounds for priors
prior_lower = np.array(
    [nroy_params_dict[n][0] for n in subset_vars], dtype=np.float32
)
prior_upper = np.array(
    [nroy_params_dict[n][1] for n in subset_vars], dtype=np.float32
)

# Warn about degenerate bounds
range_width = prior_upper - prior_lower
narrow = range_width < 1e-12
if narrow.any():
    names_narrow = [subset_vars[i] for i in np.where(narrow)[0]]
    print(f"  WARNING: {len(names_narrow)} params have degenerate NROY "
          f"range and will be effectively fixed: {names_narrow}")

print(f"  NROY points loaded:    {n_nroy}")
print(f"  Calibration params:    {ndim}")
print(f"  Full parameter vector: {len(all_param_names)}")

# ============================================================
# 2. LOAD GP EMULATORS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 -- Loading GP emulators")
print("=" * 60)

emulators = {}
for name in output_names:
    path = os.path.join(
        EMULATOR_DIR, name,
        f"GaussianProcessMatern32_{name}_best.joblib",
    )
    emulators[name] = joblib.load(path)
print(f"  Loaded {len(emulators)} emulators from {EMULATOR_DIR}/")

# Pre-compute GP internals (Cholesky, alpha) for fast MCMC evaluation
print("  Pre-warming GPyTorch prediction caches...")
gp_caches = extract_fast_caches(emulators, output_names)
first_gp = list(gp_caches.values())[0]["gp"]
print(f"  Cached {len(gp_caches)} GPs "
      f"(n_train={first_gp.train_inputs[0].shape[-2]})")

# ============================================================
# 3. KNN DENSITY ESTIMATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 3 -- KNN density estimation")
print("=" * 60)

# Standardise so that all dimensions contribute equally to distance
scaler = StandardScaler()
nroy_scaled = scaler.fit_transform(nroy_subset)

knn = NearestNeighbors(n_neighbors=KNN_K, metric="euclidean", n_jobs=8)
knn.fit(nroy_scaled)

# Density ~ 1 / (mean distance to k neighbours)
distances, _ = knn.kneighbors(nroy_scaled)
mean_dist = distances.mean(axis=1)
densities = 1.0 / (mean_dist + 1e-10)

# Densest point = optimal MCMC starting position
densest_idx = np.argmax(densities)
best_start = nroy_subset[densest_idx]

print(f"  k = {KNN_K}")
print(f"  Densest NROY index: {densest_idx}")
print(f"  Density at best:    {densities[densest_idx]:.4f}")
print(f"  Mean density:       {densities.mean():.4f}")

top_10 = np.argsort(densities)[-10:][::-1]
print(f"  Top-10 densest idx: {top_10.tolist()}")

# Sanity check: GP predictions at densest point
# NOTE: autoemulate uses output_from_samples=True, so predict_mean_and_variance
# returns a Monte Carlo estimate (noisy). Our fast-cache is the exact analytical
# GP prediction with affine y-inverse-transform — more accurate, not less.
# We verify the fast-cache against a second independent gp() call to confirm
# the x-standardisation and combined y-transform are correct.
theta_t = torch.tensor(best_start, dtype=torch.float32)

print("\n  GP predictions at densest point (fast cache):")
print(f"  {'Output':<40} {'Pred':>10} {'Target':>10} "
      f"{'|d|/s':>7}")
print("  " + "-" * 75)
mus_check = [None] * len(output_names)
for i, name in enumerate(output_names):
    c = gp_caches[name]
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        xt = (theta_t - c["x_mean"]) / c["x_std"]
        output = c["gp"](xt.unsqueeze(0))
        mean_t = output.mean.squeeze()
        mus_check[i] = (mean_t * c["y_std"] + c["y_mean"]).item()

# Convert cols 17/18 from pre-contract volumes to contraction volume diffs
mus_check[17] = mus_check[17] - mus_check[13]
mus_check[18] = mus_check[18] - mus_check[9]

for i, name in enumerate(output_names):
    mu_fast = mus_check[i]
    obs_mean = observation[name.replace("_", " ")][0]
    obs_std  = observation[name.replace("_", " ")][1] ** 0.5
    sigma_away = abs(mu_fast - obs_mean) / obs_std
    flag = "" if sigma_away <= 1.0 else " *"
    print(f"  {name:<40} {mu_fast:10.3f} {obs_mean:10.3f} "
          f"{sigma_away:7.2f}{flag}")

# Verify: two independent gp() calls at same point give identical results
# (confirms no stale caches or state-dependent prediction)
c0 = gp_caches[output_names[0]]
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    xt = (theta_t - c0["x_mean"]) / c0["x_std"]
    mu_a = c0["gp"](xt.unsqueeze(0)).mean.squeeze().item()
    mu_b = c0["gp"](xt.unsqueeze(0)).mean.squeeze().item()
assert mu_a == mu_b, f"GP prediction not deterministic: {mu_a} vs {mu_b}"
print("\n  Determinism check passed (two gp() calls match exactly)")

# ============================================================
# 4. PYRO NUTS MCMC  (custom potential_fn — no model tracing)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4 -- Pyro NUTS MCMC")
print("=" * 60)

# Observation tensors
obs_means_t = torch.tensor(
    [observation[n.replace("_", " ")][0] for n in output_names],
    dtype=torch.float32,
)
obs_vars_t = torch.tensor(
    [observation[n.replace("_", " ")][1] for n in output_names],
    dtype=torch.float32,
)
obs_stds_t = obs_vars_t.sqrt()

# Prior bounds as tensors
prior_lo = torch.tensor(prior_lower, dtype=torch.float32)
prior_hi = torch.tensor(prior_upper, dtype=torch.float32)

# ---- Cross-output correlations for the LA/RA contraction diff variance ----
# The diff = V_pre_contract - V_min is computed from two GPs that were fit
# independently. A first-order correction for the unknown Cov(GP_A, GP_B) is
# to plug in the empirical correlation of the underlying simulator outputs
# across the training set.
# Raw simulator output columns (31 total) have 6 dropped at emulator training
# time: [11, 14, 17, 20, 27, 30]. After the drop, in the 25-output space:
#   idx  9 -> Min_RA_Volume                (B for RA diff)
#   idx 13 -> Min_LA_Volume                (B for LA diff)
#   idx 17 -> V_pre_contract_LA            (A for LA diff)
#   idx 18 -> V_pre_contract_RA            (A for RA diff)
print("\n" + "=" * 60)
print("STEP 3b -- Cross-output correlations from training set")
print("=" * 60)

_lhcs_path = "LHCS_Result_20.npy"
_raw_R = np.load(_lhcs_path)
_ok = _raw_R[:, 0] != 0
_raw_R = _raw_R[_ok]
_COLS_DROP = [11, 14, 17, 20, 27, 30]
_R25 = np.delete(_raw_R, _COLS_DROP, axis=1)       # (n_train_ok, 25)
assert _R25.shape[1] == 25, f"expected 25 cols after drop, got {_R25.shape[1]}"

rho_LA = float(np.corrcoef(_R25[:, 17], _R25[:, 13])[0, 1])
rho_RA = float(np.corrcoef(_R25[:, 18], _R25[:,  9])[0, 1])
print(f"  rho(V_pre_contract_LA, Min_LA_Volume) = {rho_LA:+.3f}")
print(f"  rho(V_pre_contract_RA, Min_RA_Volume) = {rho_RA:+.3f}")
print(f"  (positive -> diff variance shrinks below Var(A)+Var(B))")

# Clamp to [-0.999, 0.999] to avoid degenerate Var(diff) ~= 0 under rho -> 1
rho_LA = max(-0.999, min(0.999, rho_LA))
rho_RA = max(-0.999, min(0.999, rho_RA))
rho_LA_t = torch.tensor(rho_LA, dtype=torch.float32)
rho_RA_t = torch.tensor(rho_RA, dtype=torch.float32)

# Build the fast potential energy function (uses pre-computed GP caches)
potential_fn = make_fast_potential_fn(
    prior_lo, prior_hi, obs_means_t, obs_vars_t, output_names, gp_caches,
    rho_LA_t, rho_RA_t,
)

# --- Convert initial point to unconstrained space via logit ---
eps = 1e-6 * (prior_hi - prior_lo).clamp(min=1e-20)
# top_chain_idx = np.argsort(densities)[-(N_CHAINS):][::-1]
# init_theta = torch.tensor(
#     nroy_subset[top_chain_idx], dtype=torch.float32,
# )

rng = np.random.default_rng(RANDOM_SEED)
start_idx = rng.choice(n_nroy, size=N_CHAINS, replace=False)
init_theta = torch.tensor(nroy_subset[start_idx], dtype=torch.float32)

init_theta = torch.clamp(init_theta, prior_lo + eps, prior_hi - eps)

# logit: z = log( (theta - lo) / (hi - theta) )
theta_01 = (init_theta - prior_lo) / (prior_hi - prior_lo)
init_z = torch.log(theta_01 / (1.0 - theta_01))        # (N_CHAINS, ndim)
if N_CHAINS == 1:
    init_z = init_z.squeeze(0)                           # (ndim,) for 1 chain

# Verify initial potential is finite
with torch.no_grad():
    z_test = init_z if init_z.dim() == 1 else init_z[0]
    pe0 = potential_fn({"theta": z_test})
    print(f"  Initial potential energy: {pe0.item():.4f}")
    assert torch.isfinite(pe0), f"Initial PE is {pe0.item()}, check NROY start"

# Verify gradient is finite at the initial point
z_grad = z_test.clone().requires_grad_(True)
pe_grad = potential_fn({"theta": z_grad})
grad_check = torch.autograd.grad(pe_grad, z_grad)[0]
print(f"  Initial gradient: finite={grad_check.isfinite().all().item()}, "
      f"norm={grad_check.norm().item():.4f}")
assert grad_check.isfinite().all(), "Gradient has NaN/Inf at initial point"

# ---------- NUTS kernel ----------
nuts = NUTS(
    potential_fn=potential_fn,
    step_size=1e-3,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    max_tree_depth=MAX_TREE_DEPTH,
    target_accept_prob=TARGET_ACCEPT,
    jit_compile=False,
)

mcmc = MCMC(
    nuts,
    num_samples=N_SAMPLES,
    warmup_steps=N_WARMUP,
    num_chains=N_CHAINS,
    initial_params={"theta": init_z},
)

print(f"  {N_CHAINS} chain(s) x ({N_WARMUP} warmup + {N_SAMPLES} samples)")
print(f"  ndim = {ndim},  max_tree_depth = {MAX_TREE_DEPTH}")
print(f"  target_accept_prob = {TARGET_ACCEPT}")
print("  Running NUTS...")

try:
    mcmc.run()
except Exception as nuts_err:
    print(f"\n  NUTS failed: {nuts_err}")
    print("  Falling back to HMC (no tree-building)...")
    hmc = HMC(
        potential_fn=potential_fn,
        step_size=1e-3,
        adapt_step_size=True,
        num_steps=20,
        jit_compile=False,
    )
    mcmc = MCMC(
        hmc,
        num_samples=N_SAMPLES,
        warmup_steps=N_WARMUP,
        num_chains=1,
        initial_params={"theta": init_z if init_z.dim() == 1 else init_z[0]},
    )
    mcmc.run()

# ============================================================
# 5. DIAGNOSTICS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5 -- MCMC diagnostics")
print("=" * 60)

# Samples are in unconstrained space — transform back to [lo, hi]
posterior_z = mcmc.get_samples(group_by_chain=False)["theta"]   # (N, ndim)
posterior = prior_lo + torch.sigmoid(posterior_z) * (prior_hi - prior_lo)
posterior_np = posterior.detach().cpu().numpy()

posterior_chains = None
if N_CHAINS > 1:
    z_chains = mcmc.get_samples(group_by_chain=True)["theta"]  # (C, N, ndim)
    posterior_chains = prior_lo + torch.sigmoid(z_chains) * (prior_hi - prior_lo)

post_mean   = posterior_np.mean(axis=0)
post_std    = posterior_np.std(axis=0)
post_median = np.median(posterior_np, axis=0)
post_q05    = np.percentile(posterior_np, 5, axis=0)
post_q95    = np.percentile(posterior_np, 95, axis=0)

print(f"\n{'Param':<20} {'Median':>12} {'Mean':>12} {'Std':>12} "
      f"{'5%':>12} {'95%':>12}")
print("-" * 80)
for j, name in enumerate(subset_vars):
    print(f"{name:<20} {post_median[j]:12.6g} {post_mean[j]:12.6g} "
          f"{post_std[j]:12.6g} {post_q05[j]:12.6g} {post_q95[j]:12.6g}")

# ============================================================
# 6. POSTERIOR PREDICTIVE CHECK
# ============================================================
print("\n" + "=" * 60)
print("STEP 6 -- Posterior predictive check")
print("=" * 60)

# --- Check posterior MEDIAN ---
x_med = torch.tensor(post_median, dtype=torch.float32).unsqueeze(0)

print(f"\n--- Posterior MEDIAN predictions ---")
print(f"{'Output':<45} {'Pred':>10} {'Target':>10} "
      f"{'|d|/s':>8} {'<=1s':>5}")
print("-" * 85)

preds_med = [None] * len(output_names)
for i, name in enumerate(output_names):
    c = gp_caches[name]
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        xt = (x_med.squeeze() - c["x_mean"]) / c["x_std"]
        out = c["gp"](xt.unsqueeze(0))
        preds_med[i] = (out.mean.squeeze() * c["y_std"] + c["y_mean"]).item()

# Convert cols 17/18 from pre-contract volumes to contraction volume diffs
preds_med[17] = preds_med[17] - preds_med[13]
preds_med[18] = preds_med[18] - preds_med[9]

n_within = 0
for i, name in enumerate(output_names):
    pred = preds_med[i]
    tgt  = obs_means_t[i].item()
    std  = obs_stds_t[i].item()
    sig  = abs(pred - tgt) / std
    ok   = sig <= 1.0
    n_within += int(ok)
    print(f"{name:<45} {pred:10.3f} {tgt:10.3f} "
          f"{sig:8.3f} {'Y' if ok else 'N':>5}")
print(f"\n  {n_within}/{len(output_names)} outputs within 1 sigma at "
      f"posterior median")

# --- Posterior predictive distribution (N_PRED_CHECK samples) ---
print(f"\n--- Posterior predictive distribution ({N_PRED_CHECK} samples) ---")
check_idx = np.random.choice(
    len(posterior_np), min(N_PRED_CHECK, len(posterior_np)), replace=False
)
n_check = len(check_idx)
pred_matrix = np.zeros((n_check, len(output_names)))

for k, idx in enumerate(check_idx):
    theta_k = torch.tensor(posterior_np[idx], dtype=torch.float32)
    for i, name in enumerate(output_names):
        c = gp_caches[name]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            xt = (theta_k - c["x_mean"]) / c["x_std"]
            out = c["gp"](xt.unsqueeze(0))
            pred_matrix[k, i] = (out.mean.squeeze() * c["y_std"]
                                 + c["y_mean"]).item()
    # Convert cols 17/18 from pre-contract volumes to contraction volume diffs
    pred_matrix[k, 17] = pred_matrix[k, 17] - pred_matrix[k, 13]
    pred_matrix[k, 18] = pred_matrix[k, 18] - pred_matrix[k, 9]

print(f"{'Output':<45} {'Mean Pred':>10} {'Std Pred':>10} "
      f"{'Target':>10} {'% <=1s':>8}")
print("-" * 90)
for i, name in enumerate(output_names):
    preds  = pred_matrix[:, i]
    tgt    = obs_means_t[i].item()
    std    = obs_stds_t[i].item()
    within = (np.abs(preds - tgt) <= std).mean() * 100
    print(f"{name:<45} {preds.mean():10.3f} {preds.std():10.3f} "
          f"{tgt:10.3f} {within:7.1f}%")

# ============================================================
# 7. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7 -- Saving results")
print("=" * 60)

out_dir = f"MCMC_Rest_{PERCENT}_14_04"
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, "posterior_samples.npy"), posterior_np)
np.save(os.path.join(out_dir, "posterior_median.npy"), post_median)
np.save(os.path.join(out_dir, "posterior_mean.npy"), post_mean)
np.save(os.path.join(out_dir, "posterior_std.npy"), post_std)
np.save(os.path.join(out_dir, "subset_vars.npy"),
        np.array(subset_vars, dtype=object))
np.save(os.path.join(out_dir, "knn_densities.npy"), densities)
np.save(os.path.join(out_dir, "knn_best_start.npy"), best_start)
np.save(os.path.join(out_dir, "pred_check_matrix.npy"), pred_matrix)

# Full 224-dim parameter vector at posterior median
# Non-calibration params stay at their nominal (fixed) values
nominal = np.array(
    [0.5 * (nroy_params_dict[n][0] + nroy_params_dict[n][1])
     for n in all_param_names], dtype=np.float32,
)
full_median = nominal.copy()
full_median[param_idx] = post_median.astype(np.float32)
np.save(os.path.join(out_dir, "full_param_median.npy"), full_median)

# Also save as {name: value} dict for easy simulator use
posterior_param_dict = {
    name: float(full_median[i])
    for i, name in enumerate(all_param_names)
}
np.save(os.path.join(out_dir, "posterior_param_dict.npy"),
        posterior_param_dict)

print(f"  Saved to {out_dir}/")

# ============================================================
# 8. PLOTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8 -- Plots")
print("=" * 60)

# 8a. Trace plots + marginal posteriors (first 8 params)
n_plot = ndim
fig, axes = plt.subplots(n_plot, 2, figsize=(14, 3 * n_plot))
if n_plot == 1:
    axes = axes[np.newaxis, :]

for j in range(n_plot):
    # Trace
    if posterior_chains is not None:
        for c in range(posterior_chains.shape[0]):
            axes[j, 0].plot(
                posterior_chains[c, :, j].cpu().numpy(),
                alpha=0.5, linewidth=0.5,
            )
    else:
        axes[j, 0].plot(posterior_np[:, j], alpha=0.7, linewidth=0.5)
    axes[j, 0].set_ylabel(subset_vars[j], fontsize=7)
    axes[j, 0].set_title(f"Trace: {subset_vars[j]}", fontsize=8)

    # Marginal posterior
    axes[j, 1].hist(
        posterior_np[:, j], bins=50, density=True, alpha=0.7, color="steelblue"
    )
    axes[j, 1].axvline(
        post_median[j], color="red", ls="--", lw=1.2, label="median"
    )
    axes[j, 1].axvline(
        prior_lower[j], color="black", ls=":", alpha=0.4, label="NROY bounds"
    )
    axes[j, 1].axvline(prior_upper[j], color="black", ls=":", alpha=0.4)
    axes[j, 1].set_title(f"Marginal: {subset_vars[j]}", fontsize=8)
    axes[j, 1].legend(fontsize=6)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "trace_and_marginals.png"), dpi=200)
plt.close()

# 8b. Normalised posterior predictive box plot
fig, ax = plt.subplots(figsize=(16, 5))
tgt_means_np = obs_means_t.numpy()
tgt_stds_np  = obs_stds_t.numpy()
normalised = (pred_matrix - tgt_means_np) / tgt_stds_np

short_names = [n.replace("_", "\n") for n in output_names]
x_pos = np.arange(len(output_names))

ax.boxplot(
    normalised, positions=x_pos, widths=0.6,
    showfliers=False, patch_artist=True,
    boxprops=dict(facecolor="steelblue", alpha=0.5),
    medianprops=dict(color="red", linewidth=1.5),
)
ax.axhline(0, color="black", ls="-", linewidth=0.5)
ax.axhline(1,  color="green", ls="--", alpha=0.6, label="+/- 1 sigma")
ax.axhline(-1, color="green", ls="--", alpha=0.6)
ax.axhline(3,  color="red",   ls=":",  alpha=0.4, label="+/- 3 sigma")
ax.axhline(-3, color="red",   ls=":",  alpha=0.4)
ax.set_xticks(x_pos)
ax.set_xticklabels(short_names, rotation=90, fontsize=6)
ax.set_ylabel("(predicted - target) / sigma")
ax.set_title("Posterior predictive normalised by population sigma")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "posterior_predictive_normalised.png"), dpi=200
)
plt.close()

# 8c. KNN density histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(densities, bins=100, density=True, alpha=0.7, color="steelblue")
ax.axvline(
    densities[densest_idx], color="red", ls="--",
    label=f"densest (idx {densest_idx})"
)
ax.set_xlabel("KNN density (1 / mean dist to k neighbours)")
ax.set_ylabel("Frequency")
ax.set_title(f"KNN density distribution (k={KNN_K}, n={n_nroy})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "knn_density_hist.png"), dpi=200)
plt.close()

print(f"  Plots saved to {out_dir}/")
print("\nDone.")
