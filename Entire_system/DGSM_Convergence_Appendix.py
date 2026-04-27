"""
DGSM_Convergence_Appendix.py

Appendix-facing post-processing for ``DGSM_Convergence.py`` outputs.

Why this script exists
----------------------
The original paper-style convergence check compared each sweep point to the
largest available sample size for a single target. That is not the best match
for the current pipeline:

1. downstream we care about the UNION of parameters selected across targets,
   not rank correlation for one output;
2. treating the final sweep as "truth" is uncomfortable if that final sweep is
   itself not demonstrably converged.

This script therefore centres the analysis on parameter discovery across ALL
targets:

* For each sweep point, build the 90%-coverage DGSM parameter set for every
  target, after first removing parameters that contribute less than 1% of that
  target's total DGSM.
* For each target, and for the global union across targets, track the
  cumulative set of parameters discovered up to that sweep point.
* Define a discovery plateau as the earliest number of base points after which
  no brand-new parameters are ever discovered again in the observed sweep.

That directly answers the appendix question:
"How many base points were needed before the union of selected parameters had
 effectively stopped growing?"

Active-set stability is still reported as a secondary diagnostic, using
consecutive Jaccard similarity. This shows whether parameter membership keeps
shuffling even after discovery has saturated.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_FILE = BASE_DIR / "DGSM_Convergence_Out" / "convergence_metrics.npz"
DEFAULT_OUT_DIR = BASE_DIR / "DGSM_Convergence_Appendix_Out"
DEFAULT_COVERAGE = 0.90
DEFAULT_JACCARD_TOL = 0.95
DEFAULT_MIN_FRAC = 0.01

OUTPUT_NAMES_REDUCED = [
    "Heart Rate", "LV Systolic Pressure", "LV Diastolic Pressure", "LV EDV", "LV ESV",
    "Max RV Volume", "Min RV Volume", "Max RV Pressure", "Min RV Pressure",
    "Min RA Volume", "Max RA Volume", "Max RA Pressure A Wave",
    "Max RA Pressure V Wave",
    "Min LA Volume", "Max LA Volume", "Max LA Pressure A Wave",
    "Max LA Pressure V Wave",
    "LA Pre-Atrial Contraction Volume", "RA Pre-Atrial Contraction Volume", "Max LV Pressure Deriv",
    "Max RV Pressure Deriv", "Tidal Volume", "Minute Ventilation",
    "PaO2", "PaCO2"
]

LEGACY_OUTPUT_NAME_MAP = {
    "Systolic Pressure": "LV Systolic Pressure",
    "Diastolic Pressure": "LV Diastolic Pressure",
    "EDV": "LV EDV",
    "ESV": "LV ESV",
    "Max RA Pressure Atrial contraction": "Max RA Pressure A Wave",
    "Max RA Pressure Tricuspid Opening": "Max RA Pressure V Wave",
    "Max LA Pressure Atrial contraction": "Max LA Pressure A Wave",
    "Max LA Pressure Mitral Opening": "Max LA Pressure V Wave",
    "LA Contraction Volume diff": "LA Pre-Atrial Contraction Volume",
    "RA Contraction Volume diff": "RA Pre-Atrial Contraction Volume",
    "LV Pressure Deriv": "Max LV Pressure Deriv",
    "RV Pressure Deriv": "Max RV Pressure Deriv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise DGSM base-point convergence across all targets using the "
            "convergence_metrics.npz file written by DGSM_Convergence.py."
        )
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_FILE,
        help="Path to convergence_metrics.npz from DGSM_Convergence.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for appendix figure and tabular outputs",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=DEFAULT_COVERAGE,
        help="Cumulative DGSM coverage used to define the selected set (default: 0.90)",
    )
    parser.add_argument(
        "--jaccard-tol",
        type=float,
        default=DEFAULT_JACCARD_TOL,
        help="Relaxed active-set stability threshold based on consecutive Jaccard similarity",
    )
    parser.add_argument(
        "--min-frac",
        type=float,
        default=DEFAULT_MIN_FRAC,
        help="Minimum per-parameter DGSM contribution, as a fraction of target total DGSM",
    )
    return parser.parse_args()


def coverage_set(dgsm_vals: np.ndarray, coverage: float, min_frac: float) -> set[int]:
    """
    Match the paper selection rule:
    keep parameters contributing at least `min_frac` of the original total DGSM,
    then take the cumulative set up to `coverage` of the original total.
    """
    values = np.asarray(dgsm_vals, dtype=float)
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    total = values.sum()
    if total <= 0:
        return set()

    order = np.argsort(values)[::-1]
    values_sorted = values[order]
    keep = values_sorted >= (min_frac * total)
    if not np.any(keep):
        return set()
    kept_order = order[keep]
    kept_values = values_sorted[keep]
    cumulative = np.cumsum(kept_values)
    cutoff = int(np.searchsorted(cumulative, coverage * total) + 1)
    cutoff = min(cutoff, len(kept_order))
    return set(kept_order[:cutoff].tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def first_plateau_from_counts(block_grid: np.ndarray, new_counts: np.ndarray) -> tuple[int, int]:
    """
    Earliest sweep point after which no further novel parameters are discovered.

    ``new_counts[s]`` is the number of parameters first discovered at sweep s.
    The returned base-point count is the smallest block count such that all later
    ``new_counts`` are zero.
    """
    for s in range(len(block_grid)):
        if np.all(new_counts[s + 1 :] == 0):
            return int(block_grid[s]), s
    return int(block_grid[-1]), len(block_grid) - 1


def first_exact_active_plateau(block_grid: np.ndarray, sets_by_step: list[set[int]]) -> tuple[int, int]:
    """Earliest sweep point after which the active selected set never changes again."""
    for s, current in enumerate(sets_by_step):
        if all(candidate == current for candidate in sets_by_step[s:]):
            return int(block_grid[s]), s
    return int(block_grid[-1]), len(block_grid) - 1


def first_relaxed_active_plateau(
    block_grid: np.ndarray,
    jaccard_series: np.ndarray,
    jaccard_tol: float,
) -> tuple[int | None, int | None]:
    """
    Earliest sweep point after which consecutive Jaccard stays above threshold.

    This is a reference-free "Cauchy-style" stability check: we only compare
    neighbouring sweep points, not to the final sweep.
    """
    for s in range(1, len(block_grid)):
        tail = jaccard_series[s:]
        if np.all(np.isfinite(tail)) and np.all(tail >= jaccard_tol):
            return int(block_grid[s]), s
    return None, None


def build_selected_sets(
    dgsm_tensor: np.ndarray,
    coverage: float,
    min_frac: float,
) -> list[list[set[int]]]:
    n_steps, n_targets, _ = dgsm_tensor.shape
    selected_sets: list[list[set[int]]] = []
    for s in range(n_steps):
        step_sets: list[set[int]] = []
        for t in range(n_targets):
            step_sets.append(coverage_set(dgsm_tensor[s, t, :], coverage, min_frac))
        selected_sets.append(step_sets)
    return selected_sets


def canonicalize_output_names(output_names: list[str]) -> list[str]:
    return [LEGACY_OUTPUT_NAME_MAP.get(name, name) for name in output_names]


def filter_to_reduced_output_set(
    output_names: list[str],
    dgsm_tensor: np.ndarray,
) -> tuple[list[str], np.ndarray, list[int]]:
    canonical_names = canonicalize_output_names(output_names)
    index_by_name: dict[str, int] = {}
    duplicates = []
    for idx, name in enumerate(canonical_names):
        if name in index_by_name:
            duplicates.append(name)
        else:
            index_by_name[name] = idx

    if duplicates:
        duplicate_str = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate canonical output names found in metrics file: {duplicate_str}")

    if all(name in index_by_name for name in OUTPUT_NAMES_REDUCED):
        keep_idx = [index_by_name[name] for name in OUTPUT_NAMES_REDUCED]
        filtered_tensor = dgsm_tensor[:, keep_idx, :]
        return OUTPUT_NAMES_REDUCED.copy(), filtered_tensor, keep_idx

    return canonical_names, dgsm_tensor, list(range(len(canonical_names)))


def compute_metrics(
    block_grid: np.ndarray,
    output_names: list[str],
    selected_sets: list[list[set[int]]],
    jaccard_tol: float,
) -> dict[str, object]:
    n_steps = len(selected_sets)
    n_targets = len(selected_sets[0])

    union_sets: list[set[int]] = []
    union_active_size = np.zeros(n_steps, dtype=int)
    union_cumulative_size = np.zeros(n_steps, dtype=int)
    union_new_count = np.zeros(n_steps, dtype=int)
    union_added = np.zeros(n_steps, dtype=int)
    union_removed = np.zeros(n_steps, dtype=int)
    union_jaccard_consec = np.full(n_steps, np.nan)

    target_active_size = np.zeros((n_steps, n_targets), dtype=int)
    target_cumulative_size = np.zeros((n_steps, n_targets), dtype=int)
    target_new_count = np.zeros((n_steps, n_targets), dtype=int)
    target_jaccard_consec = np.full((n_steps, n_targets), np.nan)

    cumulative_union: set[int] = set()
    cumulative_targets = [set() for _ in range(n_targets)]

    for s in range(n_steps):
        union_now = set().union(*selected_sets[s])
        union_sets.append(union_now)

        union_new = union_now - cumulative_union
        union_new_count[s] = len(union_new)
        cumulative_union |= union_now
        union_active_size[s] = len(union_now)
        union_cumulative_size[s] = len(cumulative_union)

        if s > 0:
            union_added[s] = len(union_now - union_sets[s - 1])
            union_removed[s] = len(union_sets[s - 1] - union_now)
            union_jaccard_consec[s] = jaccard(union_now, union_sets[s - 1])

        for t in range(n_targets):
            target_now = selected_sets[s][t]
            target_new = target_now - cumulative_targets[t]
            target_new_count[s, t] = len(target_new)
            cumulative_targets[t] |= target_now
            target_active_size[s, t] = len(target_now)
            target_cumulative_size[s, t] = len(cumulative_targets[t])

            if s > 0:
                target_jaccard_consec[s, t] = jaccard(target_now, selected_sets[s - 1][t])

    union_discovery_blocks, union_discovery_idx = first_plateau_from_counts(block_grid, union_new_count)
    union_exact_blocks, union_exact_idx = first_exact_active_plateau(block_grid, union_sets)
    union_relaxed_blocks, union_relaxed_idx = first_relaxed_active_plateau(
        block_grid, union_jaccard_consec, jaccard_tol
    )

    target_rows = []
    for t, name in enumerate(output_names):
        sets_for_target = [selected_sets[s][t] for s in range(n_steps)]
        discovery_blocks, discovery_idx = first_plateau_from_counts(block_grid, target_new_count[:, t])
        exact_blocks, exact_idx = first_exact_active_plateau(block_grid, sets_for_target)
        relaxed_blocks, relaxed_idx = first_relaxed_active_plateau(
            block_grid, target_jaccard_consec[:, t], jaccard_tol
        )
        target_rows.append(
            {
                "target": name,
                "discovery_blocks": discovery_blocks,
                "discovery_idx": discovery_idx,
                "exact_blocks": exact_blocks,
                "exact_idx": exact_idx,
                "relaxed_blocks": relaxed_blocks,
                "relaxed_idx": relaxed_idx,
                "final_active_size": target_active_size[-1, t],
                "total_discovered_size": target_cumulative_size[-1, t],
            }
        )

    target_rows.sort(
        key=lambda row: (
            row["discovery_blocks"],
            row["exact_blocks"],
            float("inf") if row["relaxed_blocks"] is None else row["relaxed_blocks"],
            row["target"],
        )
    )
    sorted_target_idx = [output_names.index(row["target"]) for row in target_rows]

    return {
        "union_sets": union_sets,
        "union_active_size": union_active_size,
        "union_cumulative_size": union_cumulative_size,
        "union_new_count": union_new_count,
        "union_added": union_added,
        "union_removed": union_removed,
        "union_jaccard_consec": union_jaccard_consec,
        "target_active_size": target_active_size,
        "target_cumulative_size": target_cumulative_size,
        "target_new_count": target_new_count,
        "target_jaccard_consec": target_jaccard_consec,
        "union_discovery_blocks": union_discovery_blocks,
        "union_discovery_idx": union_discovery_idx,
        "union_exact_blocks": union_exact_blocks,
        "union_exact_idx": union_exact_idx,
        "union_relaxed_blocks": union_relaxed_blocks,
        "union_relaxed_idx": union_relaxed_idx,
        "target_rows": target_rows,
        "sorted_target_idx": sorted_target_idx,
    }


def save_target_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    fieldnames = [
        "target",
        "discovery_blocks",
        "exact_blocks",
        "relaxed_blocks",
        "final_active_size",
        "total_discovered_size",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def save_summary_text(
    block_grid: np.ndarray,
    rows: list[dict[str, object]],
    metrics: dict[str, object],
    jaccard_tol: float,
    coverage: float,
    min_frac: float,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("DGSM appendix convergence summary\n")
        handle.write("=" * 72 + "\n")
        handle.write(f"Sweep points: {len(block_grid)}\n")
        handle.write(f"Min / max base points: {int(block_grid[0])} / {int(block_grid[-1])}\n")
        handle.write(f"Targets analysed: {len(rows)}\n")
        handle.write(
            f"Selection rule: params >= {min_frac*100:.1f}% of target DGSM, "
            f"up to {coverage*100:.0f}% cumulative DGSM\n"
        )
        handle.write(f"Relaxed consecutive Jaccard threshold: {jaccard_tol:.2f}\n\n")

        handle.write("Global union across targets\n")
        handle.write("-" * 72 + "\n")
        handle.write(
            f"Discovery plateau blocks     : {metrics['union_discovery_blocks']}\n"
        )
        handle.write(
            f"Exact active-set plateau     : {metrics['union_exact_blocks']}\n"
        )
        relaxed = metrics["union_relaxed_blocks"]
        handle.write(
            f"Relaxed active-set plateau   : {relaxed if relaxed is not None else 'not reached'}\n"
        )
        handle.write(
            f"Final active union size      : {metrics['union_active_size'][-1]}\n"
        )
        handle.write(
            f"Total discovered union size  : {metrics['union_cumulative_size'][-1]}\n\n"
        )

        handle.write("Latest-discovering targets\n")
        handle.write("-" * 72 + "\n")
        latest = sorted(rows, key=lambda row: (-row["discovery_blocks"], row["target"]))[:10]
        for row in latest:
            relaxed = "not reached" if row["relaxed_blocks"] is None else f"{row['relaxed_blocks']:>4}"
            handle.write(
                f"{row['target']:<42} "
                f"discovery={row['discovery_blocks']:>4}  "
                f"exact={row['exact_blocks']:>4}  "
                f"relaxed={relaxed}\n"
            )


def make_figure(
    block_grid: np.ndarray,
    output_names: list[str],
    metrics: dict[str, object],
    jaccard_tol: float,
    coverage: float,
    min_frac: float,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.3), constrained_layout=True)

    line_dark = "#2f4858"
    rose = "#E89A9A"
    blue = "#AFC8E8"
    teal = "#8fcad4"
    gold = "#e9c46a"
    grey = "#94a3b8"

    union_discovery_blocks = metrics["union_discovery_blocks"]
    union_exact_blocks = metrics["union_exact_blocks"]
    union_relaxed_blocks = metrics["union_relaxed_blocks"]

    step_width = int(np.median(np.diff(block_grid))) if len(block_grid) > 1 else 1
    bar_width = max(step_width * 0.65, 3)

    # Panel A: global union discovery
    ax = axes[0]
    ax.plot(block_grid, metrics["union_active_size"], color=blue, marker="o", label="Overall union size")
    ax.plot(
        block_grid,
        metrics["union_cumulative_size"],
        color="#8fcad4",
        marker="s",
        label="Cumulative discovered union size",
    )
    ax.bar(
        block_grid,
        metrics["union_new_count"],
        width=bar_width,
        color=gold,
        alpha=0.45,
        label="New parameters first discovered",
    )
    ax.axvline(
        union_discovery_blocks,
        color=teal,
        linestyle="--",
        linewidth=1.5,
        label=f"Discovery plateau = {union_discovery_blocks}",
    )
    ax.set_xlabel("Number of base points")
    ax.set_ylabel("Number of parameters")
    ax.set_title(
        f"A. Union discovery across all targets ({coverage*100:.0f}% cumulative, >= {min_frac*100:.0f}% per parameter)",
        loc="left",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=False, fontsize=9)

    # Panel B: global union active-set stability
    ax = axes[1]
    ax.axhline(0, color=grey, linewidth=1.0)
    ax.bar(
        block_grid[1:],
        metrics["union_added"][1:],
        width=bar_width,
        color="#CDB8E5",
        alpha=0.65,
        label="Added to union",
    )
    ax.bar(
        block_grid[1:],
        -metrics["union_removed"][1:],
        width=bar_width,
        color=rose,
        alpha=0.65,
        label="Removed from union",
    )
    # ax.axvline(union_discovery_blocks, color=teal, linestyle="--", linewidth=1.5)
    ax.axvline(union_exact_blocks, color=rose, linestyle=":", linewidth=1.5)
    if union_relaxed_blocks is not None:
        ax.axvline(union_relaxed_blocks, color=line_dark, linestyle="-.", linewidth=1.5)
    ax.set_xlabel("Number of base points")
    ax.set_ylabel("Parameters added / removed")
    ax.set_title("B. Active union changes between consecutive sweeps", loc="left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(
        block_grid[1:],
        metrics["union_jaccard_consec"][1:],
        color=line_dark,
        marker="o",
        linewidth=1.8,
        label="Jaccard vs previous sweep",
    )
    ax2.axhline(jaccard_tol, color=line_dark, linestyle=":", alpha=0.75, linewidth=1.5)
    ax2.set_ylabel("Jaccard similarity", color="k")
    ax2.set_ylim(0, 1.02)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("k")
    ax2.spines["right"].set_linewidth(1.0)
    ax2.tick_params(axis="y", colors="k")
    ax2.text(
        0.985,
        jaccard_tol + 0.01,
        f"Jaccard = {jaccard_tol:.2f}",
        transform=ax2.get_yaxis_transform(),
        ha="right",
        va="bottom",
        color="k",
        fontsize=9,
    )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    extra_handles = [
        # plt.Line2D([0], [0], color=teal, linestyle="--", linewidth=1.5),
        plt.Line2D([0], [0], color=rose, linestyle=":", linewidth=1.5),
    ]
    extra_labels = [
        # f"Discovery plateau = {union_discovery_blocks}",
        f"Active plateau = {union_exact_blocks}",
    ]
    if union_relaxed_blocks is not None:
        extra_handles.append(plt.Line2D([0], [0], color=line_dark, linestyle="-.", linewidth=1.5))
        extra_labels.append(f"Relaxed plateau = {union_relaxed_blocks}")
    ax.legend(
        handles1 + handles2 + extra_handles,
        labels1 + labels2 + extra_labels,
        frameon=False,
        fontsize=8.5,
        loc="lower right",
    )

    fig.suptitle(
        (
            f"DGSM convergence across all targets: parameters >= {min_frac*100:.0f}% "
            f"and cumulative contribution up to {coverage*100:.0f}%"
        ),
        fontsize=14,
    )
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.metrics_file.exists():
        raise FileNotFoundError(
            f"Could not find metrics file: {args.metrics_file}\n"
            "Run DGSM_Convergence.py first so it writes convergence_metrics.npz."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.metrics_file, allow_pickle=True)
    block_grid = np.asarray(data["block_grid"], dtype=int)
    raw_output_names = [str(name) for name in data["output_names"].tolist()]
    raw_dgsm_tensor = np.asarray(data["dgsm_tensor"], dtype=float)
    output_names, dgsm_tensor, selected_target_idx = filter_to_reduced_output_set(
        raw_output_names,
        raw_dgsm_tensor,
    )
    print(
        f"Using {len(output_names)} outputs from metrics file "
        f"({len(raw_output_names)} available before target reduction)."
    )

    selected_sets = build_selected_sets(
        dgsm_tensor,
        coverage=args.coverage,
        min_frac=args.min_frac,
    )
    metrics = compute_metrics(
        block_grid=block_grid,
        output_names=output_names,
        selected_sets=selected_sets,
        jaccard_tol=args.jaccard_tol,
    )

    csv_path = args.out_dir / "per_target_plateau_summary.csv"
    txt_path = args.out_dir / "appendix_summary.txt"
    fig_path = args.out_dir / "appendix_convergence_summary.png"
    npz_path = args.out_dir / "appendix_metrics.npz"

    save_target_csv(metrics["target_rows"], csv_path)
    save_summary_text(
        block_grid,
        metrics["target_rows"],
        metrics,
        args.jaccard_tol,
        args.coverage,
        args.min_frac,
        txt_path,
    )
    make_figure(
        block_grid,
        output_names,
        metrics,
        args.jaccard_tol,
        args.coverage,
        args.min_frac,
        fig_path,
    )

    np.savez(
        npz_path,
        block_grid=block_grid,
        output_names=np.asarray(output_names),
        source_output_names=np.asarray(raw_output_names),
        selected_target_idx=np.asarray(selected_target_idx),
        min_frac=np.asarray(args.min_frac),
        coverage=np.asarray(args.coverage),
        union_active_size=np.asarray(metrics["union_active_size"]),
        union_cumulative_size=np.asarray(metrics["union_cumulative_size"]),
        union_new_count=np.asarray(metrics["union_new_count"]),
        union_added=np.asarray(metrics["union_added"]),
        union_removed=np.asarray(metrics["union_removed"]),
        union_jaccard_consec=np.asarray(metrics["union_jaccard_consec"]),
        target_active_size=np.asarray(metrics["target_active_size"]),
        target_cumulative_size=np.asarray(metrics["target_cumulative_size"]),
        target_new_count=np.asarray(metrics["target_new_count"]),
        target_jaccard_consec=np.asarray(metrics["target_jaccard_consec"]),
        union_discovery_blocks=np.asarray(metrics["union_discovery_blocks"]),
        union_exact_blocks=np.asarray(metrics["union_exact_blocks"]),
        union_relaxed_blocks=np.asarray(
            -1 if metrics["union_relaxed_blocks"] is None else metrics["union_relaxed_blocks"]
        ),
    )

    print(f"Saved figure: {fig_path}")
    print(f"Saved target summary: {csv_path}")
    print(f"Saved text summary: {txt_path}")
    print(f"Saved numeric outputs: {npz_path}")


if __name__ == "__main__":
    main()
