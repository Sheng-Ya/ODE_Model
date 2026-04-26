import argparse
import re
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FILE_A = Path(
    r"C:\Users\vanes\Downloads\exercise_model\ODE_Exercise\Entire_system\DGSM_Rest_Paper\DGSM_20.txt"
)
DEFAULT_FILE_B = Path(
    r"C:\Users\vanes\Downloads\exercise_model\ODE_Exercise\Entire_system\DGSM_Rest_Paper\DGSM_50_tidal.txt"
)
DEFAULT_OUTPUT_PREFIX = Path(
    r"C:\Users\vanes\Downloads\exercise_model\ODE_Exercise\Entire_system\DGSM_Rest_Paper\dgsm_target_overlap_20_vs_50"
)

OUTPUT_HEADER = re.compile(r"^Output:\s*(.+)$")
PARAM_LINE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*[-+0-9\.eE]+\s*\(")

DEFAULT_TARGET_NAME_MAP = {
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


def parse_sensitivity_file(path: Path, name_map=None):
    output_params = defaultdict(set)
    current_output = None
    name_map = name_map or {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            header_match = OUTPUT_HEADER.search(line)
            if header_match:
                current_output = name_map.get(header_match.group(1).strip(), header_match.group(1).strip())
                continue

            param_match = PARAM_LINE.match(line)
            if param_match and current_output is not None:
                output_params[current_output].add(param_match.group(1))

    return dict(output_params)


def wrap_label(label: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False, break_on_hyphens=False))


def get_selected_outputs(output_params_a, output_params_b, include_targets="common"):
    outputs_a = set(output_params_a)
    outputs_b = set(output_params_b)
    if include_targets == "union":
        return sorted(outputs_a | outputs_b)
    return sorted(outputs_a & outputs_b)


def build_rows(output_params_a, output_params_b, include_targets="common"):
    outputs = get_selected_outputs(output_params_a, output_params_b, include_targets=include_targets)
    rows = []

    for output in outputs:
        params_a = output_params_a.get(output, set())
        params_b = output_params_b.get(output, set())
        shared = params_a & params_b
        only_a = params_a - params_b
        only_b = params_b - params_a
        union = params_a | params_b

        rows.append(
            {
                "target": output,
                "count_a": len(params_a),
                "count_b": len(params_b),
                "shared": len(shared),
                "only_a": len(only_a),
                "only_b": len(only_b),
                "union": len(union),
                "jaccard": len(shared) / len(union) if union else np.nan,
                "retention_a_in_b": len(shared) / len(params_a) if params_a else np.nan,
                "retention_b_in_a": len(shared) / len(params_b) if params_b else np.nan,
            }
        )

    rows.sort(key=lambda row: (-row["jaccard"], -row["shared"], -row["union"], row["target"]))
    return rows


def build_overall_summary(output_params_a, output_params_b, include_targets="common"):
    outputs = get_selected_outputs(output_params_a, output_params_b, include_targets=include_targets)
    params_a = set()
    params_b = set()

    for output in outputs:
        params_a.update(output_params_a.get(output, set()))
        params_b.update(output_params_b.get(output, set()))

    shared = params_a & params_b
    only_a = params_a - params_b
    only_b = params_b - params_a
    union = params_a | params_b

    return {
        "count_a": len(params_a),
        "count_b": len(params_b),
        "shared": len(shared),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "union": len(union),
        "jaccard": len(shared) / len(union) if union else np.nan,
        "retention_a_in_b": len(shared) / len(params_a) if params_a else np.nan,
        "retention_b_in_a": len(shared) / len(params_b) if params_b else np.nan,
    }


def add_count_labels(ax, y, left, values, color):
    for ypos, xstart, value in zip(y, left, values):
        if value < 2:
            continue
        ax.text(
            xstart + value / 2,
            ypos,
            f"{value}",
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            fontweight="bold",
        )


def make_figure(rows, overall_summary, label_a, label_b, output_png: Path):
    targets = [wrap_label(row["target"]) for row in rows]
    shared = np.array([row["shared"] for row in rows])
    only_a = np.array([row["only_a"] for row in rows])
    only_b = np.array([row["only_b"] for row in rows])
    jaccard = np.array([row["jaccard"] for row in rows])
    union = np.array([row["union"] for row in rows])

    y = np.arange(len(rows))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig_height = max(10.0, 0.34 * len(rows) + 4.0)
    fig = plt.figure(figsize=(13.5, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.45, 0.95],
        height_ratios=[1.0, max(len(rows) * 0.34, 5.0)],
    )
    ax2 = fig.add_subplot(grid[0, :])
    ax0 = fig.add_subplot(grid[1, 0])
    ax1 = fig.add_subplot(grid[1, 1], sharey=ax0)

    shared_color = "#CDB8E5"
    a_only_color = "#AFC8E8"
    b_only_color = "#E89A9A"
    line_color = "#cbd5e1"
    marker_color = "#1f2937"

    ax0.barh(y, shared, color=shared_color, height=0.72, label="Shared")
    ax0.barh(y, only_a, left=shared, color=a_only_color, height=0.72, label=f"Unique to {label_a}")
    ax0.barh(y, only_b, left=shared + only_a, color=b_only_color, height=0.72, label=f"Unique to {label_b}")

    add_count_labels(ax0, y, np.zeros_like(shared), shared, "#333333")
    add_count_labels(ax0, y, shared, only_a, "#333333")
    add_count_labels(ax0, y, shared + only_a, only_b, "#333333")

    # for ypos, total in zip(y, union):
    #     ax0.text(total + 0.35, ypos, f"n={total}", va="center", fontsize=8, color="#475569")

    ax0.set_yticks(y)
    ax0.set_yticklabels(targets)
    ax0.invert_yaxis()
    ax0.set_xlabel("Number of sensitive parameters")
    # ax0.set_title("Overlap composition by target", loc="left")
    ax0.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax0.set_axisbelow(True)
    ax0.legend(frameon=False, loc="upper right")

    ax1.hlines(y, 0, jaccard, color=line_color, linewidth=2.2)
    ax1.scatter(jaccard, y, s=38, color=marker_color, zorder=3)
    # ax1.axvline(np.nanmedian(jaccard), color="#94a3b8", linestyle="--", linewidth=1.2)
    # ax1.text(
    #     np.nanmedian(jaccard) + 0.01,
    #     -0.9,
    #     f"median = {np.nanmedian(jaccard):.2f}",
    #     fontsize=9,
    #     color="#475569",
    # )
    ax1.set_xlim(0, 1.02)
    ax1.set_xlabel(r"Jaccard similarity, $|A \cap B| / |A \cup B|$")
    # ax1.set_title("Set similarity by target", loc="left")
    ax1.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="y", left=False, labelleft=False)

    overall_shared = np.array([overall_summary["shared"]])
    overall_only_a = np.array([overall_summary["only_a"]])
    overall_only_b = np.array([overall_summary["only_b"]])
    overall_union = overall_summary["union"]
    overall_y = np.array([0])

    ax2.barh(overall_y, overall_shared, color=shared_color, height=0.36)
    ax2.barh(overall_y, overall_only_a, left=overall_shared, color=a_only_color, height=0.36)
    ax2.barh(overall_y, overall_only_b, left=overall_shared + overall_only_a, color=b_only_color, height=0.36)

    add_count_labels(ax2, overall_y, np.zeros_like(overall_shared), overall_shared, "#333333")
    add_count_labels(ax2, overall_y, overall_shared, overall_only_a, "#333333")
    add_count_labels(ax2, overall_y, overall_shared + overall_only_a, overall_only_b, "#333333")

    ax2.text(
        overall_union + 1.0,
        0,
        f"Jaccard = {overall_summary['jaccard']:.2f}",
        va="center",
        fontsize=9,
        color="#475569",
    )
    ax2.set_yticks(overall_y)
    ax2.set_yticklabels(["Union across all targets"])
    ax2.set_xlabel("Number of sensitive parameters")
    # ax2.set_title("Overall overlap across all targets", loc="left")
    ax2.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.set_xlim(0, overall_union + 12)
    ax2.set_ylim(-0.35, 0.35)

    # summary_text = (
    #     f"Jaccard = {overall_summary['jaccard']:.2f}\n"
    #     f"{label_a} retained in {label_b}: {overall_summary['retention_a_in_b']:.1%}\n"
    #     f"{label_b} retained in {label_a}: {overall_summary['retention_b_in_a']:.1%}"
    # )
    # ax2.text(
    #     0.995,
    #     0.06,
    #     summary_text,
    #     transform=ax2.transAxes,
    #     ha="right",
    #     va="bottom",
    #     fontsize=9,
    #     color="#475569",
    #     bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"},
    # )

    # fig.suptitle(
    #     f"Target-wise overlap of DGSM-sensitive parameters under {label_a} and {label_b} parameter bounds",
    #     fontsize=13,
    #     fontweight="bold",
    # )

    fig.savefig(output_png, dpi=300, bbox_inches="tight")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a journal-style target-level overlap plot for two DGSM sensitivity text files."
    )
    parser.add_argument("--file-a", type=Path, default=DEFAULT_FILE_A)
    parser.add_argument("--file-b", type=Path, default=DEFAULT_FILE_B)
    parser.add_argument("--label-a", default="+/-20%")
    parser.add_argument("--label-b", default="+/-50%")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--include-targets",
        choices=["common", "union"],
        default="common",
        help="Plot only matched targets present in both files, or the full union.",
    )
    parser.add_argument(
        "--disable-default-name-map",
        action="store_true",
        help="Disable the built-in target-name harmonization between current 20%% and 50%% rest-paper files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    name_map = {} if args.disable_default_name_map else DEFAULT_TARGET_NAME_MAP
    output_params_a = parse_sensitivity_file(args.file_a, name_map=name_map)
    output_params_b = parse_sensitivity_file(args.file_b, name_map=name_map)
    rows = build_rows(output_params_a, output_params_b, include_targets=args.include_targets)
    overall_summary = build_overall_summary(output_params_a, output_params_b, include_targets=args.include_targets)

    output_png = args.output_prefix.with_suffix(".png")
    make_figure(rows, overall_summary, args.label_a, args.label_b, output_png)

    print(f"Saved figure: {output_png}")

if __name__ == "__main__":
    main()
