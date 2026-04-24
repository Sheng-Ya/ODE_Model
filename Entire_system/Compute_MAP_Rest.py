"""Compute the maximum a posteriori (MAP) estimate from a KNN_MCMC_Rest run,
push it through the GP emulators, and plot the predictions vs. targets.

Loads posterior_samples.npy, posterior_z.npy, log_posterior_trace.npy, and the
prior bounds from a MCMC_Rest_* directory, then reports the highest-density
draw across all chains in two parametrizations:

  - theta-space MAP (default, what you usually want): argmax of
      log p(theta|data) = log_posterior_trace - log_det_jac(z)
    where log_det_jac is the sigmoid reparam Jacobian that NUTS absorbs.
  - z-space MAP: raw argmax of log_posterior_trace.

The theta-space MAP (and, for comparison, the posterior median) are then fed
to the GP emulators under EMULATOR_DIR from config.json.  The resulting
predictive means + GP stds are saved and overlaid on the posterior predictive
boxplot so you can see where the point estimate lands relative to the
observation targets and to the full posterior predictive distribution.

Usage
-----
    python Compute_MAP_Rest.py MCMC_Rest_20_21_04_3000_logspline_copula_prior

    # skip the emulator forward pass + plot (e.g. on a machine without torch):
    python Compute_MAP_Rest.py <run_dir> --no-plot

Outputs (written into run_dir):
    map_sample.npy              (d,)      theta-space MAP, constrained
    map_sample_z.npy            (d,)      same draw in unconstrained z
    map_info.json                         chain/draw indices, top-k, per-chain
    map_predictions.npy         (n_out,)  emulator mean at MAP
    map_prediction_stds.npy     (n_out,)  emulator std at MAP (GP epistemic)
    map_vs_targets.png                    normalised residual plot with
                                          MAP + median overlays on the
                                          posterior predictive boxplot.
"""

import argparse
import json
import os

import numpy as np


# ----------------------------------------------------------------------
# MAP computation (pure numpy, no torch dependency)
# ----------------------------------------------------------------------

def _log_det_jac(z, prior_lo, prior_hi):
    """Sigmoid reparam log|dtheta/dz|, summed over dims, per sample.

    log|dtheta_i/dz_i| = log(hi_i - lo_i) + logsigmoid(z_i) + logsigmoid(-z_i)
    """
    log_width = np.log(prior_hi - prior_lo)                 # (d,)
    log_sig_pos = -np.logaddexp(0.0, -z)                    # log sigmoid(z)
    log_sig_neg = -np.logaddexp(0.0,  z)                    # log sigmoid(-z)
    return (log_sig_pos + log_sig_neg + log_width).sum(axis=-1)


def compute_map(run_dir, top_k=10):
    posterior = np.load(os.path.join(run_dir, "posterior_samples.npy"))
    posterior_z = np.load(os.path.join(run_dir, "posterior_z.npy"))
    log_post_z = np.load(os.path.join(run_dir, "log_posterior_trace.npy"))
    prior_lo = np.load(os.path.join(run_dir, "prior_lower.npy"))
    prior_hi = np.load(os.path.join(run_dir, "prior_upper.npy"))

    assert posterior.shape == posterior_z.shape
    assert posterior.shape[0] == log_post_z.shape[0]

    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        n_chains = int(cfg["n_chains"])
        n_samples = int(cfg["n_samples"])
    else:
        chains = np.load(os.path.join(run_dir, "posterior_chains.npy"))
        n_chains, n_samples, _ = chains.shape
    assert n_chains * n_samples == log_post_z.shape[0]

    log_det = _log_det_jac(posterior_z, prior_lo, prior_hi)
    log_post_theta = log_post_z - log_det

    def _summarise(logp, label):
        k = int(np.argmax(logp))
        return {
            "space": label, "flat_idx": k,
            "chain": k // n_samples, "draw": k % n_samples,
            "log_post": float(logp[k]),
        }

    map_theta = _summarise(log_post_theta, "theta")
    map_z     = _summarise(log_post_z,     "z")

    topk_idx = np.argsort(log_post_theta)[::-1][:top_k]
    top_k_list = [
        {"rank": r + 1,
         "log_post_theta": float(log_post_theta[i]),
         "log_post_z":     float(log_post_z[i]),
         "chain": int(i // n_samples), "draw": int(i % n_samples),
         "flat_idx": int(i)}
        for r, i in enumerate(topk_idx)
    ]

    per_chain = []
    for c in range(n_chains):
        s, e = c * n_samples, (c + 1) * n_samples
        lp = log_post_theta[s:e]
        idx = int(np.argmax(lp))
        per_chain.append({
            "chain": c, "best_draw": idx,
            "best_log_post_theta": float(lp[idx]),
            "mean_log_post_theta": float(lp.mean()),
            "std_log_post_theta":  float(lp.std()),
        })

    return {
        "map_theta": {**map_theta,
                      "sample":   posterior[map_theta["flat_idx"]],
                      "sample_z": posterior_z[map_theta["flat_idx"]]},
        "map_z":     {**map_z,
                      "sample":   posterior[map_z["flat_idx"]],
                      "sample_z": posterior_z[map_z["flat_idx"]]},
        "top_k": top_k_list, "per_chain": per_chain,
        "n_chains": n_chains, "n_samples": n_samples,
    }


# ----------------------------------------------------------------------
# Emulator forward pass (mirrors KNN_MCMC_Rest.extract_fast_caches)
# ----------------------------------------------------------------------

def _resolve_emulator_dir(run_dir, cfg):
    """Find EMULATOR_DIR on disk. config.json stores it relative to the
    Entire_system/ cwd that KNN_MCMC_Rest.py was run from; run_dir sits in
    that same directory, so joining against run_dir's parent is robust.
    """
    emu_cfg = cfg.get("emulator_dir", "Emulator_wave_1wave")
    if os.path.isabs(emu_cfg) and os.path.isdir(emu_cfg):
        return emu_cfg
    candidates = [
        emu_cfg,
        os.path.join(os.path.dirname(os.path.abspath(run_dir)), emu_cfg),
        os.path.join(run_dir, emu_cfg),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"Could not find emulator directory '{emu_cfg}'. Tried: {candidates}. "
        f"Pass --emulator-dir to override."
    )


def _load_gp_caches(emulator_dir, output_names):
    """Load each TransformedEmulator joblib and extract the bits needed to
    evaluate a single test point: gp, x-transform, composed y-transform.

    Mirrors extract_fast_caches() in KNN_MCMC_Rest.py so that predictions
    here match what the MCMC pipeline saw.
    """
    import joblib
    import torch
    import gpytorch

    caches = {}
    for name in output_names:
        path = os.path.join(
            emulator_dir, name,
            f"GaussianProcessMatern32_{name}_best.joblib",
        )
        te = joblib.load(path)
        gp = te.model
        gp.eval()
        gp.likelihood.eval()

        x_mean = te.x_transforms[0].mean.detach().squeeze(0)
        x_std  = te.x_transforms[0].std.detach().squeeze(0)

        te_y_mean = te.y_transforms[0].mean.detach().squeeze()
        te_y_std  = te.y_transforms[0].std.detach().squeeze()

        if gp.y_transform is not None and getattr(gp.y_transform,
                                                  "_is_fitted", False):
            gp_y_mean = gp.y_transform.mean.detach().squeeze()
            gp_y_std  = gp.y_transform.std.detach().squeeze()
            y_std  = gp_y_std * te_y_std
            y_mean = gp_y_mean * te_y_std + te_y_mean
        else:
            y_std, y_mean = te_y_std, te_y_mean

        # Pre-warm Cholesky.
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _ = gp(gp.train_inputs[0][:1])

        caches[name] = {
            "gp": gp,
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std,
        }
    return caches


def _predict_at(theta_np, gp_caches, output_names):
    """Return (mu, sigma) arrays of shape (n_out,) at a single theta."""
    import torch
    import gpytorch

    theta = torch.tensor(theta_np, dtype=torch.float32)
    mu = np.zeros(len(output_names))
    sd = np.zeros(len(output_names))
    for i, name in enumerate(output_names):
        c = gp_caches[name]
        xt = (theta - c["x_mean"]) / c["x_std"]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            out = c["gp"](xt.unsqueeze(0))
            mean_t = out.mean.squeeze()
            var_t  = out.variance.squeeze().clamp(min=1e-10)
        mu[i] = (mean_t * c["y_std"] + c["y_mean"]).item()
        sd[i] = (torch.sqrt(var_t) * c["y_std"]).item()
    return mu, sd


# ----------------------------------------------------------------------
# Plot: MAP + median predictions vs targets, overlaid on posterior predictive
# ----------------------------------------------------------------------

def _plot_vs_targets(run_dir, output_names,
                     obs_means, obs_stds,
                     map_mu, map_sd,
                     median_mu=None, median_sd=None,
                     pred_matrix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_out = len(output_names)
    x_pos = np.arange(n_out)
    short_names = [n.replace("_", "\n") for n in output_names]

    fig, ax = plt.subplots(figsize=(max(12, 0.6 * n_out), 6))

    # Background: posterior predictive distribution (if available).
    if pred_matrix is not None:
        normalised = (pred_matrix - obs_means) / obs_stds
        ax.boxplot(
            normalised, positions=x_pos, widths=0.5,
            showfliers=False, patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.35,
                          edgecolor="steelblue"),
            medianprops=dict(color="steelblue", linewidth=1.2),
            whiskerprops=dict(color="steelblue", alpha=0.6),
            capprops=dict(color="steelblue", alpha=0.6),
        )

    # Target bands.
    ax.axhline(0,  color="black", ls="-",  linewidth=0.6)
    ax.axhline(1,  color="green", ls="--", alpha=0.6, label=r"$\pm 1\sigma$")
    ax.axhline(-1, color="green", ls="--", alpha=0.6)
    ax.axhline(3,  color="red",   ls=":",  alpha=0.4, label=r"$\pm 3\sigma$")
    ax.axhline(-3, color="red",   ls=":",  alpha=0.4)

    # MAP marker with GP-epistemic error bar, both in target-sigma units.
    map_res = (map_mu - obs_means) / obs_stds
    map_err = map_sd / obs_stds
    ax.errorbar(
        x_pos, map_res, yerr=map_err,
        fmt="*", color="crimson", markersize=11, markeredgecolor="black",
        markeredgewidth=0.4, elinewidth=1.0, capsize=2.5, zorder=5,
        label="MAP (GP std as error bar)",
    )

    # Posterior median marker (no error bar — keeps the plot readable).
    if median_mu is not None:
        med_res = (median_mu - obs_means) / obs_stds
        ax.scatter(
            x_pos, med_res,
            marker="D", s=30, color="navy", edgecolor="white",
            linewidth=0.5, zorder=4, label="Posterior median",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_ylabel(r"(predicted - target) / $\sigma_{\mathrm{obs}}$")
    ax.set_title("MAP emulator predictions vs. targets "
                 "(normalised by observation sigma)")

    # y-range: clip extreme outliers so MAP stays visible.
    ymin = min(-3.5, np.nanpercentile(map_res - map_err, 5),
               np.nanmin(map_res - map_err) - 0.5)
    ymax = max( 3.5, np.nanpercentile(map_res + map_err, 95),
               np.nanmax(map_res + map_err) + 0.5)
    ax.set_ylim(ymin, ymax)

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(run_dir, "map_vs_targets.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# ----------------------------------------------------------------------
# Reporting + save
# ----------------------------------------------------------------------

def _print_report(result, run_dir, top_k):
    mt, mz = result["map_theta"], result["map_z"]
    print(f"Run dir:      {run_dir}")
    print(f"Chains:       {result['n_chains']} x {result['n_samples']} draws "
          f"(N = {result['n_chains'] * result['n_samples']})")
    print()
    print("MAP (theta-space, log p(theta|data)):")
    print(f"  log p        = {mt['log_post']:.4f}")
    print(f"  chain / draw = {mt['chain']} / {mt['draw']}  "
          f"(flat index {mt['flat_idx']})")
    print()
    print("MAP (z-space, log p(z|data) = potential's negative):")
    print(f"  log p        = {mz['log_post']:.4f}")
    print(f"  chain / draw = {mz['chain']} / {mz['draw']}  "
          f"(flat index {mz['flat_idx']})")
    if mt["flat_idx"] != mz["flat_idx"]:
        print("  NOTE: theta-space and z-space MAPs differ — posterior has "
              "mass near the NROY-box edges where the sigmoid Jacobian is "
              "large.")
    print()

    print("Per-chain best (theta-space):")
    print(f"  {'chain':>5}  {'best_draw':>9}  {'best_logp':>11}  "
          f"{'mean_logp':>11}  {'std_logp':>9}")
    for row in result["per_chain"]:
        print(f"  {row['chain']:>5d}  {row['best_draw']:>9d}  "
              f"{row['best_log_post_theta']:>11.4f}  "
              f"{row['mean_log_post_theta']:>11.4f}  "
              f"{row['std_log_post_theta']:>9.4f}")
    best_logps = [r["best_log_post_theta"] for r in result["per_chain"]]
    spread = max(best_logps) - min(best_logps)
    print(f"Chain best-logp spread: {spread:.4f} nats")
    if spread > 5.0:
        print("  WARN: >5 nats between chain peaks — chains may not be mixing "
              "around the same mode. Check split-R-hat in "
              "mcmc_diagnostics.json.")
    print()

    print(f"Top {top_k} draws (theta-space):")
    print(f"  {'rank':>4}  {'chain':>5}  {'draw':>6}  "
          f"{'logp_theta':>12}  {'logp_z':>12}")
    for row in result["top_k"]:
        print(f"  {row['rank']:>4d}  {row['chain']:>5d}  {row['draw']:>6d}  "
              f"{row['log_post_theta']:>12.4f}  {row['log_post_z']:>12.4f}")


def _print_prediction_table(output_names, obs_means, obs_stds, map_mu, map_sd):
    print()
    print("MAP predictions vs. targets (|res|/sigma > 1 flagged):")
    print(f"  {'Output':<45} {'MAP pred':>10} {'GP std':>8} "
          f"{'Target':>10} {'Res/sig':>8} {'Flag':>5}")
    print("-" * 90)
    n_within = 0
    for i, name in enumerate(output_names):
        res_over_sig = abs(map_mu[i] - obs_means[i]) / obs_stds[i]
        ok = res_over_sig <= 1.0
        n_within += int(ok)
        flag = "" if ok else "!"
        print(f"  {name:<45} {map_mu[i]:>10.3f} {map_sd[i]:>8.3f} "
              f"{obs_means[i]:>10.3f} {res_over_sig:>8.3f} {flag:>5}")
    print(f"  {n_within}/{len(output_names)} outputs within 1 sigma at MAP.")


def _save_map(result, run_dir):
    np.save(os.path.join(run_dir, "map_sample.npy"),
            result["map_theta"]["sample"])
    np.save(os.path.join(run_dir, "map_sample_z.npy"),
            result["map_theta"]["sample_z"])

    info = {
        "map_theta": {k: v for k, v in result["map_theta"].items()
                      if k not in ("sample", "sample_z")},
        "map_z":     {k: v for k, v in result["map_z"].items()
                      if k not in ("sample", "sample_z")},
        "top_k":     result["top_k"],
        "per_chain": result["per_chain"],
        "n_chains":  result["n_chains"],
        "n_samples": result["n_samples"],
    }
    with open(os.path.join(run_dir, "map_info.json"), "w") as f:
        json.dump(info, f, indent=2)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir",
                   help="Path to a MCMC_Rest_* output directory.")
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of highest-density draws to report.")
    p.add_argument("--emulator-dir", default=None,
                   help="Override EMULATOR_DIR from config.json.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip emulator forward pass and plot (report only).")
    p.add_argument("--no-save", action="store_true",
                   help="Print only; don't write any files.")
    args = p.parse_args()

    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"run_dir not found: {args.run_dir}")

    # ---- 1. MAP computation + textual report ------------------------
    result = compute_map(args.run_dir, top_k=args.top_k)
    _print_report(result, args.run_dir, args.top_k)
    if not args.no_save:
        _save_map(result, args.run_dir)
        print()
        print(f"Saved: {os.path.join(args.run_dir, 'map_sample.npy')}")
        print(f"Saved: {os.path.join(args.run_dir, 'map_sample_z.npy')}")
        print(f"Saved: {os.path.join(args.run_dir, 'map_info.json')}")

    if args.no_plot:
        return

    # ---- 2. Emulator forward pass at MAP (+ median for comparison) --
    cfg_path = os.path.join(args.run_dir, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)

    output_names = np.load(os.path.join(args.run_dir, "output_names.npy"),
                           allow_pickle=True).tolist()
    obs_means = np.load(os.path.join(args.run_dir, "obs_means.npy"))
    obs_vars  = np.load(os.path.join(args.run_dir, "obs_vars.npy"))
    obs_stds  = np.sqrt(obs_vars)

    if args.emulator_dir is not None:
        emulator_dir = args.emulator_dir
        if not os.path.isdir(emulator_dir):
            raise SystemExit(f"--emulator-dir not found: {emulator_dir}")
    else:
        emulator_dir = _resolve_emulator_dir(args.run_dir, cfg)
    print()
    print(f"Loading emulators from: {emulator_dir}")

    gp_caches = _load_gp_caches(emulator_dir, output_names)
    print(f"  Loaded {len(gp_caches)} GPs.")

    map_sample = result["map_theta"]["sample"]
    map_mu, map_sd = _predict_at(map_sample, gp_caches, output_names)

    median_mu = median_sd = None
    median_path = os.path.join(args.run_dir, "posterior_median.npy")
    if os.path.exists(median_path):
        median_sample = np.load(median_path)
        median_mu, median_sd = _predict_at(median_sample, gp_caches,
                                           output_names)

    _print_prediction_table(output_names, obs_means, obs_stds, map_mu, map_sd)

    if not args.no_save:
        np.save(os.path.join(args.run_dir, "map_predictions.npy"),     map_mu)
        np.save(os.path.join(args.run_dir, "map_prediction_stds.npy"), map_sd)

    # ---- 3. Plot ----------------------------------------------------
    pred_matrix = None
    pred_path = os.path.join(args.run_dir, "pred_check_matrix.npy")
    if os.path.exists(pred_path):
        pred_matrix = np.load(pred_path)

    out_png = _plot_vs_targets(
        args.run_dir, output_names, obs_means, obs_stds,
        map_mu, map_sd,
        median_mu=median_mu, median_sd=median_sd,
        pred_matrix=pred_matrix,
    )
    print()
    print(f"Saved: {out_png}")
    if not args.no_save:
        print(f"Saved: {os.path.join(args.run_dir, 'map_predictions.npy')}")
        print(f"Saved: {os.path.join(args.run_dir, 'map_prediction_stds.npy')}")


if __name__ == "__main__":
    main()
