import argparse
import ast
import sys
import warnings
from pathlib import Path

import numpy as np
from SALib import ProblemSpec


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT = SCRIPT_DIR / "Derivative-based GSA Union.py"
X_FILE = SCRIPT_DIR / "DGSM_500_X_union_50_27_05.npy"
RESULT_FILE = SCRIPT_DIR / "DGSM_500_Result_union_50_27_05.npy"

# dgsm_edited.py lives one level above this script.
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
import dgsm_edited as dgsm  # noqa: E402


warnings.filterwarnings(
    "ignore",
    message=".*use_inf_as_na option is deprecated.*",
    category=FutureWarning,
)


# Parameters in both files
SUBSET_VARS = {
    "a2", "ahead1", "C2", "C_O2_param1", "C_O2_param2", "C_sv", "Emax_lv0",
    "Emax_ra", "Emax_rv0", "fab_o", "fes_o", "fev_inf", "fev_o", "GT_v",
    "KE_la", "KE_lv", "KE_ra", "KE_rv", "l", "MO2_bp", "P0_la", "P0_lv",
    "P0_rv", "PaCO2_n", "PAMO2_nominal", "r", "R_po", "R_rs", "R_sa",
    "R_tr", "rise_time_ven", "s", "scale_param1", "scale_param4", "T0",
    "theta_po_max", "theta_tr_max", "V0_dead", "V_nominal", "V_scale",
    "Vu_ev0", "Vu_jp", "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "Vu_sv0",
}

# Only in Rest
SUBSET_REST_ONLY = {
    "Emax_la", "f_ab_max", "fes_inf", "fes_min", "Io_sv", "K2", "kcc_sv",
    "kes", "P_n", "R_pp", "rise_time_atr", "Wb_sh",
}

# Only in Exercise
SUBSET_EXERCISE_ONLY = {
    "C_pv", "E_rs", "G_ap", "GEmax_lv", "GEmax_rv", "GR_amp", "GT_s",
    "GV_sv", "KcCO2", "P_n_max", "phi_max", "Rvc_n", "VA_rest", "Wp_v",
    "Yv_max",
}

REST_PARAMETER_SET = SUBSET_VARS | SUBSET_REST_ONLY
EXERCISE_PARAMETER_SET = SUBSET_VARS | SUBSET_EXERCISE_ONLY


def _assigned_names(node):
    names = set()
    for target in getattr(node, "targets", []):
        if isinstance(target, ast.Name):
            names.add(target.id)
    return names


def _source_tree(source_script):
    return ast.parse(source_script.read_text(encoding="utf-8"), filename=str(source_script))


def load_problem_spec(source_script=SOURCE_SCRIPT):
    """Load the original union ProblemSpec without running the original script."""
    tree = _source_tree(source_script)
    needed = {"percentage", "lower", "upper", "sp"}
    nodes = []

    for node in tree.body:
        if isinstance(node, ast.Assign) and _assigned_names(node) & needed:
            nodes.append(node)
            if "sp" in _assigned_names(node):
                break

    if not any(isinstance(node, ast.Assign) and "sp" in _assigned_names(node) for node in nodes):
        raise RuntimeError(f"Could not find the ProblemSpec assignment in {source_script}")

    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"ProblemSpec": ProblemSpec}
    exec(compile(module, str(source_script), "exec"), namespace, namespace)
    return namespace["sp"]


def load_output_names(source_script=SOURCE_SCRIPT):
    """Load the target labels from the original script without running analyses or plotting."""
    tree = _source_tree(source_script)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and "output_names" in _assigned_names(node)
    ]

    if len(nodes) < 2:
        raise RuntimeError(f"Could not find the output_names assignments in {source_script}")

    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_script), "exec"), namespace, namespace)
    return list(namespace["output_names"])


def load_and_filter_data(apply_filters=True):
    X = np.load(X_FILE)
    result = np.load(RESULT_FILE)

    if X.shape[0] != result.shape[0]:
        raise RuntimeError(f"X rows ({X.shape[0]}) do not match Result rows ({result.shape[0]})")

    D = X.shape[1]
    block_size = D + 1

    if X.shape[0] % block_size != 0:
        raise RuntimeError(
            f"X has {X.shape[0]} rows, which is not divisible by block size {block_size}"
        )

    if not apply_filters:
        return X, result, X.shape[0] // block_size, X.shape[0] // block_size

    n_blocks = X.shape[0] // block_size
    base_idx = np.arange(0, X.shape[0], block_size)

    mask_blocks = result[base_idx, 0] != 0
    mask_blocks_nan = np.array([
        np.all(np.isfinite(result[i:i + block_size]))
        for i in base_idx
    ])

    block_std = np.zeros((n_blocks, result.shape[1]))
    for b, i in enumerate(base_idx):
        block_std[b] = np.nanstd(result[i:i + block_size], axis=0)

    std_mean = np.nanmean(block_std, axis=0)
    std_std = np.nanstd(block_std, axis=0)
    std_thresh = std_mean + 3 * std_std
    mask_blocks_std = np.all(block_std <= std_thresh, axis=1)

    HR_col = 0
    mask_blocks_conv = np.array([
        np.all(np.abs(result[i + 1:i + block_size, HR_col] - result[i, HR_col]) < 0.03)
        for i in base_idx
    ])

    tidal_col = 25
    mask_blocks_conv_tidal = np.array([
        np.all(np.abs(result[i + 1:i + block_size, tidal_col] - result[i, tidal_col]) < 0.03)
        for i in base_idx
    ])

    mask_blocks = (
        mask_blocks
        & mask_blocks_nan
        & mask_blocks_conv
        & mask_blocks_std
        & mask_blocks_conv_tidal
    )
    mask_full = np.repeat(mask_blocks, block_size)
    return X[mask_full], result[mask_full], n_blocks, int(np.count_nonzero(mask_blocks))


def selected_set_for_output(output_label):
    if output_label.startswith("Rest "):
        return REST_PARAMETER_SET, "union of overlap and rest parameters"
    if output_label.startswith("Exercise "):
        return EXERCISE_PARAMETER_SET, "union of overlap and exercise parameters"
    raise ValueError(f"Cannot infer rest/exercise parameter set for output: {output_label}")


def validate_parameter_sets(problem_names):
    problem_name_set = set(problem_names)
    missing = sorted((REST_PARAMETER_SET | EXERCISE_PARAMETER_SET) - problem_name_set, key=str.lower)
    if missing:
        raise RuntimeError(
            "These requested parameters are not in the union ProblemSpec: "
            + ", ".join(missing)
        )


def format_output_block(output_label, names, dgsm_values, selected_names, selected_label):
    finite = np.isfinite(dgsm_values) & (dgsm_values >= 0)
    total = np.sum(dgsm_values[finite])
    if total <= 0 or not np.isfinite(total):
        return [
            "",
            "=" * 80,
            f"Output: {output_label}",
            f"Skipping {output_label} (non-positive/invalid DGSM total: {total})",
        ]

    selected_mask = np.isin(names, list(selected_names)) & finite
    selected_dgsm = dgsm_values[selected_mask]
    selected_param_names = names[selected_mask]

    if selected_dgsm.size != len(selected_names):
        found = set(selected_param_names)
        missing = sorted(set(selected_names) - found, key=str.lower)
        raise RuntimeError(
            f"{output_label}: {len(missing)} selected parameters were not returned by DGSM: "
            + ", ".join(missing)
        )

    order = np.argsort(selected_dgsm)[::-1]
    selected_dgsm = selected_dgsm[order]
    selected_param_names = selected_param_names[order]

    reached = selected_dgsm.sum() / total
    min_idx = int(np.argmin(selected_dgsm))
    min_dgsm = selected_dgsm[min_idx]
    min_percent = min_dgsm / total * 100

    lines = [
        "",
        "=" * 80,
        f"Output: {output_label}",
        f"Min per-parameter contribution: {min_percent:.3f}% (DGSM >= {min_dgsm:.4e})",
        f"Parameters selected: {len(selected_param_names)} ({selected_label})",
        f"Fraction of total DGSM reached: {reached * 100:.2f}%",
        "-" * 80,
    ]

    for name, value in zip(selected_param_names, selected_dgsm):
        lines.append(f"{name:25s} : {value:.4e}  ({value / total * 100:.3f}%)")

    return lines


def build_report(num_resamples=2, apply_filters=True):
    sp = load_problem_spec()
    output_names = load_output_names()
    X, result, total_blocks, kept_blocks = load_and_filter_data(apply_filters=apply_filters)

    if X.shape[1] != sp["num_vars"]:
        raise RuntimeError(f"X has {X.shape[1]} columns but ProblemSpec has {sp['num_vars']} variables")
    if result.shape[1] != len(output_names):
        raise RuntimeError(
            f"Result has {result.shape[1]} output columns but {len(output_names)} names were loaded"
        )

    problem_names = np.asarray(sp["names"])
    validate_parameter_sets(problem_names)

    lines = [
        f"Loaded X: {X_FILE}",
        f"Loaded Result: {RESULT_FILE}",
        f"Blocks kept after existing quality filters: {kept_blocks}/{total_blocks}",
        f"Rest parameters selected: {len(REST_PARAMETER_SET)}",
        f"Exercise parameters selected: {len(EXERCISE_PARAMETER_SET)}",
    ]

    for col, output_label in enumerate(output_names):
        Y = result[:, col]
        if not np.all(np.isfinite(Y)):
            lines.extend([
                "",
                "=" * 80,
                f"Output: {output_label}",
                f"Skipping {output_label} (non-finite output values)",
            ])
            continue

        Si = dgsm.analyze(sp, X, Y, num_resamples=num_resamples, print_to_console=False)
        dgsm_values = np.asarray(Si["dgsm"], dtype=float)
        names = np.asarray(Si["names"])
        selected_names, selected_label = selected_set_for_output(output_label)
        lines.extend(format_output_block(output_label, names, dgsm_values, selected_names, selected_label))

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Report how much total DGSM is covered by the fixed rest/exercise "
            "parameter unions for every union DGSM target."
        )
    )
    parser.add_argument(
        "--num-resamples",
        type=int,
        default=2,
        help=(
            "Bootstrap resamples passed to dgsm_edited.analyze (default: 2). "
            "The printed coverage uses raw DGSM, not dgsm_conf."
        ),
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip the same block quality filters used by Derivative-based GSA Union.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional text file path to receive the same report printed to stdout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_report(num_resamples=args.num_resamples, apply_filters=not args.no_filter)
    print(report)
    if args.output is not None:
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
