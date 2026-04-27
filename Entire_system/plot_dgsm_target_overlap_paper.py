import argparse
import re
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
DETAIL_PARAM_LINE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*([-+0-9\.eE]+)\s*\(([0-9\.]+)%\)")

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


def parse_sensitivity_details(path: Path, name_map=None):
    output_details = defaultdict(dict)
    current_output = None
    name_map = name_map or {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            header_match = OUTPUT_HEADER.search(line)
            if header_match:
                current_output = name_map.get(header_match.group(1).strip(), header_match.group(1).strip())
                continue

            param_match = DETAIL_PARAM_LINE.match(line)
            if param_match and current_output is not None:
                param = param_match.group(1)
                output_details[current_output][param] = {
                    "dgsm": float(param_match.group(2)),
                    "pct": float(param_match.group(3)),
                }

    return dict(output_details)


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
    union_a = set()
    for output in outputs:
        union_a.update(output_params_a.get(output, set()))
    rows = []

    for output in outputs:
        params_a = output_params_a.get(output, set())
        params_b = output_params_b.get(output, set())
        shared = params_a & params_b
        only_a = params_a - params_b
        only_b = params_b - params_a
        only_b_in_union_a = {param for param in only_b if param in union_a}
        only_b_new_global = only_b - only_b_in_union_a
        union = params_a | params_b

        rows.append(
            {
                "target": output,
                "count_a": len(params_a),
                "count_b": len(params_b),
                "shared": len(shared),
                "only_a": len(only_a),
                "only_b": len(only_b),
                "only_b_in_union_a": len(only_b_in_union_a),
                "only_b_new_global": len(only_b_new_global),
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


def build_new_parameter_contribution_data(
    output_params_a,
    output_params_b,
    output_details_b,
    rows,
    include_targets="common",
):
    outputs = get_selected_outputs(output_params_a, output_params_b, include_targets=include_targets)
    params_a = set()
    params_b = set()

    for output in outputs:
        params_a.update(output_params_a.get(output, set()))
        params_b.update(output_params_b.get(output, set()))

    new_params = sorted(params_b - params_a)
    parameter_totals = {param: 0.0 for param in new_params}
    target_rows = []

    for row in rows:
        target = row["target"]
        detail = output_details_b.get(target, {})
        contributions = {}
        for param in new_params:
            pct = detail.get(param, {}).get("pct")
            if pct is not None:
                contributions[param] = pct
                parameter_totals[param] += pct

        target_rows.append(
            {
                "target": target,
                "contributions": contributions,
                "sum_pct": sum(contributions.values()),
            }
        )

    ordered_params = sorted(new_params, key=lambda param: (-parameter_totals[param], param))
    return {
        "new_params": ordered_params,
        "parameter_totals": parameter_totals,
        "target_rows": target_rows,
    }


def sort_rows_by_added_contribution(rows, contribution_data):
    contribution_map = {item["target"]: item["sum_pct"] for item in contribution_data["target_rows"]}
    rows.sort(
        key=lambda row: (
            -contribution_map.get(row["target"], 0.0),
            -row["jaccard"],
            -row["shared"],
            -row["union"],
            row["target"],
        )
    )
    return rows


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


def make_figure(rows, overall_summary, contribution_data, label_a, label_b, output_png: Path):
    targets = [wrap_label(row["target"]) for row in rows]
    shared = np.array([row["shared"] for row in rows])
    only_a = np.array([row["only_a"] for row in rows])
    only_b_in_union_a = np.array([row["only_b_in_union_a"] for row in rows])
    only_b_new_global = np.array([row["only_b_new_global"] for row in rows])

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

    top_height = 1.2
    bottom_height = max(len(rows) * 0.34, 5.6)
    fig_height = top_height + bottom_height + 1.8
    fig = plt.figure(figsize=(13.5, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.45, 0.95],
        height_ratios=[top_height, bottom_height],
    )
    ax_top = fig.add_subplot(grid[0, :])
    ax_left = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax_left)

    shared_color = "#CDB8E5"
    a_only_color = "#AFC8E8"
    b_only_existing_color = "#8fcad4"
    b_only_new_color = "#E89A9A"

    overall_shared = np.array([overall_summary["shared"]])
    overall_only_a = np.array([overall_summary["only_a"]])
    overall_only_b = np.array([overall_summary["only_b"]])
    overall_union = overall_summary["union"]
    overall_y = np.array([0])

    ax_top.barh(overall_y, overall_shared, color=shared_color, height=0.36)
    ax_top.barh(overall_y, overall_only_a, left=overall_shared, color=a_only_color, height=0.36)
    ax_top.barh(overall_y, overall_only_b, left=overall_shared + overall_only_a, color=b_only_new_color, height=0.36)

    add_count_labels(ax_top, overall_y, np.zeros_like(overall_shared), overall_shared, "#333333")
    add_count_labels(ax_top, overall_y, overall_shared, overall_only_a, "#333333")
    add_count_labels(ax_top, overall_y, overall_shared + overall_only_a, overall_only_b, "#333333")

    ax_top.text(
        overall_union + 1.0,
        0,
        (
            f"Jaccard = {overall_summary['jaccard']:.2f}\n"
            "|A ∩ B| / |A ∪ B|"
        ),
        va="center",
        fontsize=9,
        color="#475569",
    )

    ax_top.set_yticks(overall_y)
    ax_top.set_yticklabels(["Union across all targets"])
    ax_top.set_xlabel("Number of sensitive parameters")
    ax_top.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax_top.set_axisbelow(True)
    ax_top.set_xlim(0, overall_union + 12)
    ax_top.set_ylim(-0.35, 0.35)

    ax_left.barh(y, shared, color=shared_color, height=0.72, label="Shared")
    ax_left.barh(y, only_a, left=shared, color=a_only_color, height=0.72, label=f"Unique to {label_a}")
    ax_left.barh(
        y,
        only_b_in_union_a,
        left=shared + only_a,
        color=b_only_existing_color,
        height=0.72,
        label=f"Unique to {label_b}, but in {label_a} union",
    )
    ax_left.barh(
        y,
        only_b_new_global,
        left=shared + only_a + only_b_in_union_a,
        color=b_only_new_color,
        height=0.72,
        label=f"Unique to {label_b}",
    )

    add_count_labels(ax_left, y, np.zeros_like(shared), shared, "#333333")
    add_count_labels(ax_left, y, shared, only_a, "#333333")
    add_count_labels(ax_left, y, shared + only_a, only_b_in_union_a, "#333333")
    add_count_labels(ax_left, y, shared + only_a + only_b_in_union_a, only_b_new_global, "#333333")

    ax_left.set_yticks(y)
    ax_left.set_yticklabels(targets)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("Number of sensitive parameters")
    ax_left.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax_left.set_axisbelow(True)
    ax_left.legend(frameon=False, loc="upper right")

    contribution_rows = contribution_data["target_rows"]
    contribution_sums = np.array([item["sum_pct"] for item in contribution_rows])
    background = np.full(len(contribution_rows), 100.0)

    ax_right.barh(y, background, color="#e5e7eb", height=0.54, label="Total DGSM (normalised)")

    # Monochrome palette for the +/-50%-only DGSM contributions in panel C.
    light_rose = np.array(mcolors.to_rgb("#F8DCDC"))
    dark_rose = np.array(mcolors.to_rgb("#C96A6A"))
    positive_widths = []
    for item in contribution_rows:
        positive_widths.extend([value for value in item["contributions"].values() if value > 0])
    min_width = min(positive_widths) if positive_widths else 0.0
    max_width = max(positive_widths) if positive_widths else 1.0

    for ypos, item in zip(y, contribution_rows):
        left = 0.0
        ordered_contributions = sorted(item["contributions"].items(), key=lambda pair: (-pair[1], pair[0]))
        for _, width in ordered_contributions:
            if max_width > min_width:
                t = np.clip((width - min_width) / (max_width - min_width), 0.0, 1.0)
            else:
                t = 0.5
            color = mcolors.to_hex(light_rose * (1 - t) + dark_rose * t)
            ax_right.barh(
                ypos,
                width,
                left=left,
                color=color,
                height=0.54,
                edgecolor="#4b5563",
                linewidth=0.2,
            )
            left += width

    for ypos, total in zip(y, contribution_sums):
        if total > 0:
            ax_right.text(total + 1.0, ypos, f"{total:.1f}%", va="center", fontsize=8, color="#475569")

    ax_right.set_xlim(0, 100)
    ax_right.set_xlabel("Normalised total DGSM (%)")
    ax_right.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax_right.set_axisbelow(True)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.text(
        0.98,
        1.02,
        "Darker rose = larger DGSM contribution",
        transform=ax_right.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#475569",
    )

    # legend_handles = [plt.Rectangle((0, 0), 1, 1, color="#e5e7eb", ec="none")]
    # legend_labels = ["Total DGSM (normalised)"]
    # for param in new_params:
    #     legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=param_colors[param], ec="none"))
    #     legend_labels.append(param)
    # ax_right.legend(
    #     legend_handles,
    #     legend_labels,
    #     frameon=False,
    #     ncol=3,
    #     fontsize=8,
    #     loc="upper left",
    #     bbox_to_anchor=(0, 1.12),
    #     columnspacing=1.0,
    #     handlelength=1.4,
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
    output_details_b = parse_sensitivity_details(args.file_b, name_map=name_map)
    rows = build_rows(output_params_a, output_params_b, include_targets=args.include_targets)
    overall_summary = build_overall_summary(output_params_a, output_params_b, include_targets=args.include_targets)
    contribution_data = build_new_parameter_contribution_data(
        output_params_a,
        output_params_b,
        output_details_b,
        rows,
        include_targets=args.include_targets,
    )
    rows = sort_rows_by_added_contribution(rows, contribution_data)
    contribution_data = build_new_parameter_contribution_data(
        output_params_a,
        output_params_b,
        output_details_b,
        rows,
        include_targets=args.include_targets,
    )

    output_png = args.output_prefix.with_suffix(".png")
    make_figure(rows, overall_summary, contribution_data, args.label_a, args.label_b, output_png)

    print(f"Saved figure: {output_png}")

if __name__ == "__main__":
    main()
