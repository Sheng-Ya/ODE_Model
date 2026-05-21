"""
Single-panel comparison of emulator predictive variance across history matching stages.

The plot shows grouped horizontal box plots of
`log10(emulator predictive variance / observation variance)` for each target.
Each stage (Initial, Wave 1, ..., Wave N) is drawn in a different colour, using
cached results under the artifact directory by default.

Example:
    python Plot_Emulator_Variance_ThreePanel.py ^
        --artifacts-dir DGSM_Rest_Paper/HM_Rest_low_RA_high_C_pa ^
        --eval-wave 4 ^
        --after-wave 3 ^
        --out emulator_variance_grouped_boxplot.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from SALib import ProblemSpec
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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


DEFAULT_TARGET_NAME_MAP = {
    "Systolic Pressure": "LV Systolic Pressure",
    "Diastolic Pressure": "LV Diastolic Pressure",
    "EDV": "LV End-Diastolic Volume",
    "ESV": "LV End-Systolic Volume",
    "Max RV Volume": "RV End-Diastolic Volume",
    "Min RV Volume": "RV End-Systolic Volume",
    "Max RV Pressure": "RV Systolic Pressure",
    "Min RV Pressure": "RV Diastolic Pressure",
    "Max RA Pressure Atrial contraction": "Max RA Pressure A Wave",
    "Max RA Pressure Tricuspid Opening": "Max RA Pressure V Wave",
    "Max LA Pressure Atrial contraction": "Max LA Pressure A Wave",
    "Max LA Pressure Mitral Opening": "Max LA Pressure V Wave",
    "LA Contraction Volume diff": "LA Pre-Atrial Contraction Volume",
    "RA Contraction Volume diff": "RA Pre-Atrial Contraction Volume",
    "LV Pressure Deriv": "Max LV Pressure Derivative",
    "RV Pressure Deriv": "Max RV Pressure Derivative",
    "PaO2": r"PaO$_2$",
    "PaCO2": r"PaCO$_2$",
}
STAGE_COLORS = [
    "#BBA3D6",
    "#9DB8D8",
    "#7DB6C0",
    "#D68484",
]


MODEL_NAME = "GaussianProcessMatern32"
FLAG_THRESHOLD = 0.1
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = SCRIPT_DIR / "DGSM_Rest_Paper" / "HM_Rest_low_RA_high_C_pa"
INITIAL_EMULATOR_DIR_NAME = "Emulator_rest_initial"

def unwrap_emulator(obj):
    if hasattr(obj, "predict_mean_and_variance"):
        return obj
    model = getattr(obj, "model", None)
    if model is not None and hasattr(model, "predict_mean_and_variance"):
        return model
    raise TypeError(f"Unsupported emulator object type: {type(obj)!r}")


def format_target_name(name: str) -> str:
    pretty_name = name.replace("_", " ")
    return DEFAULT_TARGET_NAME_MAP.get(pretty_name, pretty_name)



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
    batch_size: int,
) -> np.ndarray:

    variances = []
    for name in output_names:
        emu_path = emulator_dir / name / f"{MODEL_NAME}_{name}_best.joblib"

        emulator = unwrap_emulator(joblib.load(emu_path))
        target_variances = []
        with torch.no_grad():
            for start in range(0, x_subset.shape[0], batch_size):
                batch = x_subset[start:start + batch_size]
                _, var_pred = emulator.predict_mean_and_variance(batch)
                target_variances.append(
                    var_pred.detach().cpu().numpy().reshape(-1).astype(np.float32)
                )
        variances.append(np.concatenate(target_variances))

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


def load_eval_points(artifacts_dir: Path, eval_wave: int) -> np.ndarray:
    test_path = artifacts_dir / f"test_params_wave_{eval_wave}.npy"
    mask_path = artifacts_dir / f"nroy_mask_wave_{eval_wave}.npy"

    missing = [path for path in (test_path, mask_path) if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Expected wave {eval_wave} evaluation files in the artifact directory:\n"
            f"{missing_text}"
        )

    test_params = np.load(test_path)
    nroy_mask = np.load(mask_path).astype(bool)
    if test_params.shape[0] != nroy_mask.shape[0]:
        raise ValueError(
            f"{test_path} has {test_params.shape[0]} rows but "
            f"{mask_path} has {nroy_mask.shape[0]} entries."
        )

    return test_params[nroy_mask]


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
    batch_size: int,
) -> dict:

    if cache_path.exists():
        print(f"[{stage_key}] using cached emulator results: {cache_path}")
        var_matrix = load_cached_variance_matrix(cache_path, n_eval)
    else:
        print(f"[{stage_key}] cache miss, evaluating emulator predictions")
        var_matrix = predict_variance_matrix(emulator_dir, x_subset, batch_size)
        save_variance_matrix_cache(cache_path, stage_label, emulator_dir, var_matrix)

    return summarise_variance_matrix(var_matrix)


def plot_grouped_ratio_boxplot(
    ax,
    stage_results: list[tuple[str, dict]],
    ordered_names: list[str],
) -> None:
    n_targets = len(ordered_names)
    n_stages = len(stage_results)
    target_spacing = 3.0
    base_positions = np.arange(n_targets) * target_spacing
    total_group_height = 2
    box_width = total_group_height / max(n_stages, 1) * 0.85

    if n_stages == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(
            -total_group_height / 2 + box_width / 2,
            total_group_height / 2 - box_width / 2,
            n_stages,
        )

    legend_handles = []
    for stage_idx, (stage_label, stage_result) in enumerate(stage_results):
        stage_data = [
            np.log10(np.clip(stage_result["ratios"][name], 1e-300, None))
            for name in ordered_names
        ]
        color = STAGE_COLORS[stage_idx % len(STAGE_COLORS)]
        bp = ax.boxplot(
            stage_data,
            vert=False,
            positions=base_positions + offsets[stage_idx],
            widths=box_width,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.7)
            patch.set_linewidth(1.2)
        for median in bp["medians"]:
            median.set_color(color)
            median.set_linewidth(1.4)
        for whisker in bp["whiskers"]:
            whisker.set_color(color)
            whisker.set_linewidth(1.1)
        for cap in bp["caps"]:
            cap.set_color(color)
            cap.set_linewidth(1.1)

        legend_handles.append(
            Patch(facecolor=color, edgecolor=color, alpha=0.7, label=stage_label)
        )

    ax.axvline(np.log10(FLAG_THRESHOLD), color="k", linestyle="--", linewidth=1.5)
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=1.5)
    ax.set_yticks(base_positions)
    ax.set_yticklabels([format_target_name(name) for name in ordered_names])
    ax.set_ylim(
        base_positions[0] - total_group_height / 2 - 0.35,
        base_positions[-1] + total_group_height / 2 + 0.35,
    )
    separator_positions = (base_positions[:-1] + base_positions[1:]) / 2
    for y in separator_positions:
        ax.axhline(y, color="#c7c7c7", linewidth=0.7, zorder=0)
    ax.invert_yaxis()
    ax.set_ylabel("Target")
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")
    ax.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        frameon=True,
        facecolor="white",
        edgecolor="#d0d0d0",
        framealpha=0.9,
    )


def build_stage_series(
    artifacts_dir: Path,
    x_subset: torch.Tensor,
    n_eval: int,
    after_wave: int,
    initial_emulator_dir: Path,
    cache_dir: Path,
    eval_wave: int,
    batch_size: int,
) -> list[tuple[str, dict]]:
    stages = [(
        "Initial",
        evaluate_emulator_directory(
            stage_key="rest_initial",
            stage_label="Initial",
            emulator_dir=initial_emulator_dir,
            x_subset=x_subset,
            n_eval=n_eval,
            cache_path=build_stage_cache_path(
                cache_dir=cache_dir,
                stage_key="rest_initial",
                eval_wave=eval_wave,
            ),
            batch_size=batch_size,
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
                    batch_size=batch_size,
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
        default=DEFAULT_ARTIFACTS_DIR,
        type=Path,
        help=(
            "Directory containing test_params_wave_k.npy, nroy_mask_wave_k.npy, "
            "Emulator_rest_initial, and Emulator_wave_k snapshots."
        ),
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
        help="Include stages Initial, Wave 1, ..., Wave N in the grouped box plot.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=Path,
        help=(
            "Directory for saved emulator-variance cache files. "
            "Default: <artifacts-dir>/emulator_prediction_cache_3"
        ),
    )
    parser.add_argument(
        "--out",
        default="emulator_variance_grouped_boxplot.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Number of fourth-wave points to evaluate per emulator prediction batch.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    artifacts_dir = args.artifacts_dir
    initial_emulator_dir = artifacts_dir / INITIAL_EMULATOR_DIR_NAME
    out_path = Path(args.out)
    cache_dir = (
        args.cache_dir
        if args.cache_dir
        else artifacts_dir / "emulator_prediction_cache_3"
    )

    eval_points = load_eval_points(artifacts_dir, args.eval_wave)

    x_subset = torch.from_numpy(eval_points[:, parameter_idx]).float()
    stage_results = build_stage_series(
        artifacts_dir=artifacts_dir,
        x_subset=x_subset,
        n_eval=eval_points.shape[0],
        after_wave=args.after_wave,
        initial_emulator_dir=initial_emulator_dir,
        cache_dir=cache_dir,
        eval_wave=args.eval_wave,
        batch_size=args.batch_size,
    )

    pre_hm_lookup = {row["target"]: row for row in stage_results[0][1]["rows"]}
    ordered_names = sorted(
        output_names,
        key=lambda name: pre_hm_lookup[name]["ratio_median"],
        reverse=True,
    )

    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 26,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "axes.linewidth": 1.2,
        "lines.linewidth": 1.8,
    })

    fig, ax = plt.subplots(figsize=(18, 23))
    plot_grouped_ratio_boxplot(
        ax=ax,
        stage_results=stage_results,
        ordered_names=ordered_names,
    )
    ax.set_xlabel(r"$\log_{10}(V_{\mathrm{Emulator}} / V_{\mathrm{obs}})$")
    # ax.set_title("Grouped emulator variance box plots across stages")

    # fig.suptitle(
    #     f"Emulator variance across history matching stages "
    #     f"(evaluated on N={eval_points.shape[0]} held-out wave {args.eval_wave} NROY points)",
    #     fontsize=15,
    #     y=0.98,
    # )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {out_path}")

    write_summary_csv(out_path, stage_results, eval_points.shape[0])


if __name__ == "__main__":
    main()
