"""
Three-panel comparison of emulator predictive variance before and after history matching.

Panels:
  (a) Pre-HM emulator variance relative to observation variance, using the
      `Emulator_Paper_same_1000` emulators evaluated on held-out NROY points
      from a later wave.
  (b) Post-HM emulator variance relative to observation variance, using the
      saved emulator snapshot after a specified wave (default: wave 3).
  (c) Per-output median emulator predictive variance / observation variance across
      stages: Pre-HM, Wave 1, ..., selected post-HM wave.

Recommended usage for a 4-wave history matching run:
  - Evaluate on `test_params_wave_4.npy` filtered by `nroy_mask_wave_4.npy`
  - Compare `Emulator_Paper_same_1000` against `Emulator_wave_3`
  - Plot stage trajectory for Pre-HM, Wave 1, Wave 2, Wave 3

Example:
    python Plot_Emulator_Variance_ThreePanel.py ^
        --artifacts-dir . ^
        --eval-wave 4 ^
        --after-wave 3 ^
        --out emulator_variance_three_panel.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from SALib import ProblemSpec
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

sp = ProblemSpec({
    "names": [
        "beta2", "C2", "K2", "a2", "alpha2", "KCCO2", "GV_dead",
        "KcCO2", "KcMRV", "KpCO2", "KpO2", "V0_dead", "VA_rest",
        "E_rs", "R_rs",
        "C_jp", "C_sa", "L_sa", "R_sa", "C_amv", "C_bv", "C_ev", "C_hv",
        "C_rmv", "C_sv", "kr_am", "P_0", "R_amv_n", "R_bv_n", "R_ev_n", "R_hv_n",
        "R_rmv_n", "R_sv_n", "K1_vc", "D1", "Vvc_min", "Kr_vc",
        "Rvc_n", "C_pa", "C_pp", "C_pv", "L_pa", "R_pa", "R_pp",
        "R_pv", "Emax_la", "P0_la", "Emax_ra", "P0_ra", "KE_la", "KE_ra", "P0_lv",
        "P0_rv", "s",
        "fab_o", "fes_o", "fes_inf", "fes_max", "fev_o", "fev_inf", "kes", "kev",
        "Io_sh", "Io_sp", "Io_sv", "Io_v", "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v",
        "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min", "Ysv_max", "Ysv_min", "Yv_max", "Yv_min",
        "theta_v", "Wb_sh", "Wb_sp", "Wb_sv", "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v",
        "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
        "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp",
        "GR_sp", "GV_amv", "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0",
        "R_sp0", "g_ccsh", "g_ccsp", "kisc_sh", "kisc_sp", "kisc_sv",
        "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn", "theta_spn", "theta_svn", "x_sh", "x_sp",
        "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min", "k_ab", "P_n", "P_n_max", "f_acCO2_n",
        "f_ac_max", "f_ac_min", "k_ac", "K_H", "PaO2_ac_n", "G_ap", "GT_s", "GT_v",
        "T0", "A", "B", "C", "D", "Cvb_O2_n", "gb_O2", "MO2_bp",
        "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2", "grm_O2", "Kh_CO2", "Krm_CO2", "MO2_hpn",
        "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n", "gam_O2", "gM", "Io_met", "kmet",
        "MO2_ampn", "phi_max", "phi_min",
        "Kp_ao", "Kf_ao", "Kb_ao", "Kv_ao", "theta_ao_max",
        "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max",
        "Kp_po", "Kf_po", "Kb_po", "Kv_po", "theta_po_max",
        "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max",
        "alpha_O2", "R_po", "R_mi", "R_tr", "R_ao",
        "C_O2_param1", "C_O2_param2", "C_O2_param3", "PAMO2_nominal",
        "Vu_bv", "Vu_hv", "Vu_jp", "Vu_vc",
        "Vu_pp", "Vu_pv", "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv",
        "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp", "tau_Rep", "tau_Rrmp", "tau_Rsp", "tau_Vamv",
        "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0", "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc",
        "tau_isc", "tau_p", "tau_z", "tau_ac", "tau_ap", "tau_Ts", "tau_Tv", "tau_CO2",
        "tau_O2", "tau_w", "tau_M", "tau_met", "DEmax_lv", "DEmax_rv", "DR_amp", "DR_ep",
        "DR_rmp", "DR_sp", "DV_amv", "DV_ev", "DV_rmv", "DV_sv", "DT_s", "DT_v",
        "Dmet", "Ta", "KE_lv", "KE_rv", "T1", "T2", "VL_CO2", "VL_O2",
        "KCSFCO2", "VB", "tauMR", "VTCO2", "VTO2", "tau_MRV",
        "scale_param1", "scale_param3", "scale_param4", "scale_param6", "Pa_O2_lower",
        "rise_time_atr", "rise_time_ven", "fall_time_ven", "ahead1",
        "theta_min", "r", "l", "V_nominal", "V_scale",
    ],
    # bounds aren't needed for variance analysis; dummies keep SALib happy if used
    "bounds": [[0.0, 1.0]] * 272,  # placeholder; not used here
})

subset_vars_set = {
    "a2", "ahead1", "beta2", "C2", "C_jp", "C_O2_param1", "C_sv", "Cvam_O2_n", "E_rs", "Emax_la",
    "Emax_lv0", "Emax_ra", "Emax_rv0", "f_ab_max", "fab_o", "fall_time_ven", "fes_inf", "fes_min",
    "fes_o", "fev_inf", "fev_o", "GT_s", "GT_v", "Io_met", "Io_sv", "K2", "k_ab", "kcc_sv", "KE_la",
    "KE_lv", "KE_ra", "KE_rv", "kes", "kmet", "Kv_mi", "Kv_po", "Kv_tr", "l", "MO2_bp", "P0_la", "P0_lv",
    "P0_ra", "P0_rv", "P_n", "PaCO2_n", "r", "R_pa", "R_pp", "R_rs", "R_sa", "rise_time_atr",
    "rise_time_ven", "Rvc_n", "T0", "theta_svn", "V0_dead", "V_nominal", "V_scale", "Vu_amv0", "Vu_bv",
    "Vu_ev0", "Vu_jp", "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "Vu_sv0", "Wb_sh", "Wb_sv",
}
subset_vars = [name for name in sp["names"] if name in subset_vars_set]
parameter_idx = [sp["names"].index(name) for name in subset_vars]

output_names = [
    "LA_Contraction_Volume_diff", "RA_Contraction_Volume_diff",
    "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV",
    "ESV", "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure",
    "Min_RV_Pressure", "Min_RA_Volume", "Max_RA_Volume", "Max_RA_Pressure_Atrial_contraction",
    "Max_RA_Pressure_Tricuspid_Opening", "Min_LA_Volume", "Max_LA_Volume", "Max_LA_Pressure_Atrial_contraction",
    "Max_LA_Pressure_Mitral_Opening", "LV_Pressure_Deriv",
    "RV_Pressure_Deriv", "Tidal_Volume", "Minute_Ventilation", "PaO2", "PaCO2",
]

observation = {
    "Heart_Rate": (1.23, 0.05), "Systolic_Pressure": (123, 324),
    "Diastolic_Pressure": (76.7, 65.61), "EDV": (152.1, 767.29),
    "ESV": (62.3, 243.36), "Max_RV_Volume": (151.9, 1004.89),
    "Min_RV_Volume": (64.4, 299.29), "Max_RV_Pressure": (22.5, 56.25),
    "Min_RV_Pressure": (4.0, 9.0), "Min_RA_Volume": (45.7, 125.44), "Max_RA_Volume": (92.4, 380.25),
    "Max_RA_Pressure_Atrial_contraction": (8.0, 9.0),
    "Max_RA_Pressure_Tricuspid_Opening": (5.0, 9.0),
    "Min_LA_Volume": (30.6, 84.64), "Max_LA_Volume": (68.3, 306.25),
    "Max_LA_Pressure_Atrial_contraction": (13.0, 9.0),
    "Max_LA_Pressure_Mitral_Opening": (12.0, 9.0),
    # Exact propagated pre-A moments implied by
    #   V_pre = V_min + f * (V_max - V_min),  f ~ N(0.25, 0.0002777)
    "LA_Contraction_Volume_diff": (40.025, 67.253867386),
    "RA_Contraction_Volume_diff": (57.375, 95.071688266),
    "LV_Pressure_Deriv": (1461.0, 146689.0),
    "RV_Pressure_Deriv": (271.0, 3025.0),
    "Tidal_Volume": (0.850, 0.16),
    "Minute_Ventilation": (11.4, 15.21),
    "PaO2": (102.3, 125.44), "PaCO2": (35.5, 24.01),
}


MODEL_NAME = "GaussianProcessMatern32"
FLAG_THRESHOLD = 0.1
RAW_EMULATOR_DIR = "Emulator_Paper_same_1000"

def unwrap_emulator(obj):
    if hasattr(obj, "predict_mean_and_variance"):
        return obj
    model = getattr(obj, "model", None)
    if model is not None and hasattr(model, "predict_mean_and_variance"):
        return model
    raise TypeError(f"Unsupported emulator object type: {type(obj)!r}")



def build_stage_cache_path(
    cache_dir: Path,
    stage_key: str,
    eval_wave: int,
) -> Path:
    parts = [stage_key, f"evalwave_{eval_wave}"]
    return cache_dir / ("__".join(parts) + ".npz")


def predict_variance_matrix(
    emulator_dir: Path,
    x_subset: torch.Tensor,
) -> np.ndarray:

    variances = []
    for name in output_names:
        emu_path = emulator_dir / name / f"{MODEL_NAME}_{name}_best.joblib"

        emulator = unwrap_emulator(joblib.load(emu_path))
        with torch.no_grad():
            _, var_pred = emulator.predict_mean_and_variance(x_subset)
        variances.append(var_pred.detach().cpu().numpy().reshape(-1).astype(np.float32))

    return np.stack(variances, axis=0)


def load_cached_variance_matrix(cache_path: Path, expected_n_eval: int) -> np.ndarray:
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached emulator results: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)
    cached_output_names = [str(name) for name in data["output_names"]]
    if cached_output_names != output_names:
        raise ValueError(
            f"{cache_path} was saved with a different output ordering."
        )

    var_matrix = np.asarray(data["var_matrix"], dtype=np.float32)
    if var_matrix.shape != (len(output_names), expected_n_eval):
        raise ValueError(
            f"{cache_path} has shape {var_matrix.shape}, expected "
            f"({len(output_names)}, {expected_n_eval})."
        )
    return var_matrix


def save_variance_matrix_cache(
    cache_path: Path,
    stage_label: str,
    emulator_dir: Path,
    var_matrix: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        stage_label=stage_label,
        emulator_dir=str(emulator_dir),
        model_name=MODEL_NAME,
        output_names=np.array(output_names),
        var_matrix=np.asarray(var_matrix, dtype=np.float32),
        n_eval=np.int64(var_matrix.shape[1]),
    )


def summarise_variance_matrix(var_matrix: np.ndarray) -> dict:
    rows = []
    all_ratios = {}

    for idx, name in enumerate(output_names):
        var_code = np.asarray(var_matrix[idx], dtype=float)
        v_obs = float(observation[name][1])
        ratio = var_code / v_obs

        q05, q50, q95 = np.quantile(ratio, [0.05, 0.5, 0.95])
        rows.append({
            "target": name,
            "V_obs": v_obs,
            "V_code_mean": float(np.mean(var_code)),
            "V_code_median": float(np.median(var_code)),
            "V_code_q05": float(np.quantile(var_code, 0.05)),
            "V_code_q95": float(np.quantile(var_code, 0.95)),
            "ratio_mean": float(np.mean(ratio)),
            "ratio_median": float(q50),
            "ratio_q05": float(q05),
            "ratio_q95": float(q95),
            "median_var_over_obs_var": float(np.median(ratio)),
        })
        all_ratios[name] = ratio

    return {
        "rows": rows,
        "ratios": all_ratios,
    }


def evaluate_emulator_directory(
    stage_key: str,
    stage_label: str,
    emulator_dir: Path,
    x_subset: torch.Tensor,
    n_eval: int,
    cache_path: Path,
) -> dict:

    if cache_path.exists():
        print(f"[{stage_key}] using cached emulator results: {cache_path}")
        var_matrix = load_cached_variance_matrix(cache_path, n_eval)
    else:
        print(f"[{stage_key}] cache miss, evaluating emulator predictions")
        var_matrix = predict_variance_matrix(emulator_dir, x_subset)
        save_variance_matrix_cache(cache_path, stage_label, emulator_dir, var_matrix)

    return summarise_variance_matrix(var_matrix)


def plot_ratio_boxplot(
    ax,
    stage_result: dict,
    ordered_names: list[str],
    title: str,
    show_ylabels: bool,
) -> None:
    data = [
        np.log10(np.clip(stage_result["ratios"][name], 1e-300, None))
        for name in ordered_names
    ]
    bp = ax.boxplot(
        data,
        vert=False,
        labels=ordered_names if show_ylabels else [""] * len(ordered_names),
        showfliers=False,
        patch_artist=True,
    )

    row_lookup = {row["target"]: row for row in stage_result["rows"]}
    for patch, name in zip(bp["boxes"], ordered_names):
        med_ratio = row_lookup[name]["ratio_median"]
        patch.set_facecolor("#d9534f" if med_ratio > FLAG_THRESHOLD else "#5cb85c")
        patch.set_alpha(0.6)

    ax.axvline(np.log10(FLAG_THRESHOLD), color="k", linestyle="--", linewidth=1.5)
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=1.5)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")
    if show_ylabels:
        ax.set_ylabel("Output")
    else:
        ax.tick_params(axis="y", length=0)


def build_stage_series(
    artifacts_dir: Path,
    x_subset: torch.Tensor,
    n_eval: int,
    after_wave: int,
    raw_emulator_dir: Path,
    cache_dir: Path,
    eval_wave: int,
) -> list[tuple[str, dict]]:
    stages = [(
        "Pre-HM",
        evaluate_emulator_directory(
            stage_key="pre_hm",
            stage_label="Pre-HM",
            emulator_dir=raw_emulator_dir,
            x_subset=x_subset,
            n_eval=n_eval,
            cache_path=build_stage_cache_path(
                cache_dir=cache_dir,
                stage_key="pre_hm",
                eval_wave=eval_wave,
            ),
        ),
    )]
    for wave in range(1, after_wave + 1):
        stage_dir = artifacts_dir / f"Emulator_wave_{wave}"
        stages.append(
            (
                f"Wave {wave}",
                evaluate_emulator_directory(
                    stage_key=f"wave_{wave}",
                    stage_label=f"Wave {wave}",
                    emulator_dir=stage_dir,
                    x_subset=x_subset,
                    n_eval=n_eval,
                    cache_path=build_stage_cache_path(
                        cache_dir=cache_dir,
                        stage_key=f"wave_{wave}",
                        eval_wave=eval_wave,
                    ),
                ),
            )
        )
    return stages


def write_summary_csv(
    out_path: Path,
    stage_results: list[tuple[str, dict]],
    n_eval: int,
) -> None:
    csv_path = out_path.with_name(out_path.stem + "_summary.csv")
    with csv_path.open("w", newline="") as fh:
        fieldnames = [
            "stage",
            "n_eval",
            "target",
            "V_obs",
            "V_code_mean",
            "V_code_median",
            "V_code_q05",
            "V_code_q95",
            "ratio_mean",
            "ratio_median",
            "ratio_q05",
            "ratio_q95",
            "median_var_over_obs_var",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for stage_name, result in stage_results:
            for row in result["rows"]:
                writer.writerow({"stage": stage_name, "n_eval": n_eval, **row})
    print(f"Saved summary CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        default="HM_Rest_5",
        help="Directory containing test_params_wave_k.npy, nroy_mask_wave_k.npy, and Emulator_wave_k snapshots.",
    )
    parser.add_argument(
        "--eval-wave",
        type=int,
        default=4,
        help="Wave whose held-out NROY points are used for evaluation.",
    )
    parser.add_argument(
        "--after-wave",
        type=int,
        default=3,
        help="Use the emulator snapshot after this wave for panel (b), and include waves 1..after-wave in panel (c).",
    )
    parser.add_argument(
        "--cache-dir",
        default="emulator_prediction_cache_3",
        help="Directory for saved emulator-variance cache files. Default: emulator_prediction_cache_3",
    )
    parser.add_argument(
        "--out",
        default="emulator_variance_three_panel_test.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    raw_emulator_dir = Path(RAW_EMULATOR_DIR)
    out_path = Path(args.out)
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else artifacts_dir / "emulator_prediction_cache"
    )

    test_path = artifacts_dir / f"test_params_wave_{args.eval_wave}.npy"
    mask_path = artifacts_dir / f"nroy_mask_wave_{args.eval_wave}.npy"

    test_params = np.load(test_path)
    nroy_mask = np.load(mask_path).astype(bool)

    eval_points = test_params[nroy_mask]

    x_subset = torch.from_numpy(eval_points[:, parameter_idx]).float()
    stage_results = build_stage_series(
        artifacts_dir=artifacts_dir,
        x_subset=x_subset,
        n_eval=eval_points.shape[0],
        after_wave=args.after_wave,
        raw_emulator_dir=raw_emulator_dir,
        cache_dir=cache_dir,
        eval_wave=args.eval_wave,
    )

    pre_label, pre_result = stage_results[0]
    post_label, post_result = stage_results[-1]
    ordered_names = [
        row["target"]
        for row in sorted(pre_result["rows"], key=lambda row: row["ratio_median"])
    ]

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 1.2,
        "lines.linewidth": 1.8,
    })

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(23, 9),
        gridspec_kw={"width_ratios": [1.15, 1.15, 1.35]},
    )
    ax_a, ax_b, ax_c = axes

    plot_ratio_boxplot(
        ax=ax_a,
        stage_result=pre_result,
        ordered_names=ordered_names,
        title=f"(a) {pre_label} emulators",
        show_ylabels=True,
    )
    ax_a.set_xlabel(r"$\log_{10}(V_{\mathrm{code}} / V_{\mathrm{obs}})$")

    plot_ratio_boxplot(
        ax=ax_b,
        stage_result=post_result,
        ordered_names=ordered_names,
        title=f"(b) {post_label} emulators",
        show_ylabels=False,
    )
    ax_b.set_xlabel(r"$\log_{10}(V_{\mathrm{code}} / V_{\mathrm{obs}})$")

    stage_labels = [label for label, _ in stage_results]
    rel_var_matrix = np.array([
        [result["rows"][output_names.index(name)]["median_var_over_obs_var"] for name in output_names]
        for _, result in stage_results
    ])
    cmap = plt.get_cmap("tab20")
    for j, name in enumerate(output_names):
        ax_c.plot(
            stage_labels,
            rel_var_matrix[:, j],
            marker="o",
            color=cmap(j % 20),
            alpha=0.85,
            label=name.replace("_", " "),
        )
    ax_c.plot(
        stage_labels,
        np.nanmedian(rel_var_matrix, axis=1),
        marker="D",
        color="black",
        linewidth=3.0,
        label="median across outputs",
    )
    ax_c.axhline(FLAG_THRESHOLD, color="k", linestyle="--", linewidth=1.2)
    ax_c.axhline(1.0, color="grey", linestyle="--", linewidth=1.2)
    ax_c.set_yscale("log")
    ax_c.set_ylabel("median emulator variance / obs variance")
    ax_c.set_title("(c) Median predictive variance across stages")
    ax_c.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_c.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=False,
    )

    fig.suptitle(
        f"Emulator variance before and after history matching "
        f"(evaluated on N={eval_points.shape[0]} held-out wave {args.eval_wave} NROY points)",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {out_path}")

    write_summary_csv(out_path, stage_results, eval_points.shape[0])


if __name__ == "__main__":
    main()
