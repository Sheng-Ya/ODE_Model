"""
Hierarchical Edge Bundling Plot for DGSM Sensitivity Analysis
=============================================================
Reads DGSM_20_rest_final.txt and visualises the relationship between
target outputs and their sensitive parameters, grouped by physiological
subsystem.

- Outer ring  : parameters (grouped by subsystem)
- Inner hub   : target outputs (auto-detected from file)
- Node size   : how many targets a parameter is sensitive to
- Edge opacity: strength of sensitivity (DGSM %)
- Edge colour : target output identity

Usage:
    python edge_bundling.py DGSM_20_rest_final.txt
    python edge_bundling.py DGSM_20_rest_final.txt --output my_plot
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from collections import defaultdict, Counter, OrderedDict
from matplotlib import colors


# ── 1. PARSER ────────────────────────────────────────────────────────────────

def parse_sensitivity_file(filename):
    """
    Parse sensitivity output file.

    Returns
    -------
    output_params : dict  {output_name: {param: pct, ...}}
    param_counts  : Counter  how many outputs each parameter appears in
    """
    output_params = defaultdict(dict)
    current_output = None

    output_header = re.compile(r"^Output:\s*(.+)$")
    param_line = re.compile(
        r"^\s*([A-Za-z0-9_]+)\s*:\s*[\d.eE+\-]+\s*\(\s*([\d.]+)%\s*\)"
    )

    with open(filename, "r") as f:
        for line in f:
            header_match = output_header.search(line)
            if header_match:
                current_output = header_match.group(1).strip()
                continue

            param_match = param_line.match(line)
            if param_match and current_output is not None:
                param = param_match.group(1)
                pct = float(param_match.group(2))
                output_params[current_output][param] = pct

    param_counts = Counter()
    for params in output_params.values():
        param_counts.update(params.keys())

    return dict(output_params), param_counts


# ── 2. CONFIGURATION ─────────────────────────────────────────────────────────

GROUPS = OrderedDict([
    ("Chemoreflex", ["PaCO2_n"]),
    ("Baroreflex",   ["P_n", "f_ab_max", "k_ab", "Wb_sv", "Wb_sh", "fab_o"]),
    ("Efferent\nSympathetic/\nVagal Firing",     ["fes_min", "fes_o", "fes_inf", "fev_o", "GT_v", "T0", "fev_inf", "GT_s", "kes",]),
    # ("Efferent\nVagal",           ["fev_o", "GT_v"]),
    ("Cardiac\nMechanics",        ["Emax_lv0", "KE_lv", "KE_rv", "rise_time_ven",
                                   "ahead1", "Emax_rv0", "KE_la", "P0_rv", "fall_time_ven", "Emax_la", "P0_la", "P0_lv", "rise_time_atr",
					"Emax_ra", "KE_ra", "P0_ra"]),
    ("Pericardial\nProperties", ["V_nominal", "V_scale", "Vu_lv", "r", "l"]),
    ("Systemic\nVolumes", ["Vu_ev0", "Vu_jp", "Vu_bv", "Vu_amv0", "Vu_rv", "Vu_la", "Vu_ra", "Vu_sv0"]),
    ("Vascular\nProperties",      ["R_sa", "C_sv", "C_jp", "R_pa", "R_pp", "Rvc_n"]),
    ("Central\nCommand",          ["Io_sv", "kcc_sv", "theta_svn", "Io_met", "kmet"]),
    ("Gas\nExchange",   ["MO2_bp", "beta2", "C_O2_param1", "C2", "Cvam_O2_n", "a2", "K2"]),
    ("Valve\nDynamics",   ["Kv_po", "Kv_mi", "Kv_tr"]),
    ("Ventilation",   ["R_rs", "V0_dead", "E_rs"]),
])

TARGET_DISPLAY_NAMES = {
    "Systolic Pressure":  "Systolic\nPressure",
    "Diastolic Pressure": "Diastolic\nPressure",
    "LV Pressure Deriv":  "LV dP/dt",
    "RV Pressure Deriv": "RV dP/dt",
    "RV Systolic Pressure": "RV Systolic\nPressure",
    "RV Diastolic Pressure": "RV Diastolic\nPressure",
    "Max LA Pressure Atrial Contraction": "Max LA P\nLA Contraction",
    "Max LA Pressure Mitral Opening": "Max LA P\nMitral Opening",
    "LA Volume before LA Contraction": "V before LA\n Contracts",
    "Max LA Volume": "Max LA V",
    "Min LA Volume": "Min LA V",
    "Max RA Volume": "Max RA V",
    "Min RA Volume": "Max RA V",
    "RA Volume before RA Contraction": "V before RA\n Contracts",
    "Max RA Pressure Atrial Contraction": "Max RA P\nRA Contraction",
    "Max RA Pressure Tricuspid Opening": "Max RA P\nTricuspid Opening",
    "PaO2": "PaO$_2$",
    "PaCO2": "PaCO$_2$",

}

PARAM_DISPLAY_NAMES = {
    "rise_time_ven": "rise time\nven",
    "C_O2_param1": "C O$_2$\nparam1",
    "Cvam_O2_n": "Cvam\nO$_2$ n",
    "fall_time_ven": "fall time\nven",
    "rise_time_atr": "rise time\natr",
}

COLOUR_CYCLE = [
    "#AA003A",
    "#D87A38",
    "#2A9D8F",
    "#7209B7",
    "#2F6FB3",
    "#D62828",
    "#457B9D",
    "#6A994E",
]

GROUP_BAND_COLOURS = [
    "#E8D0D8", "#E8D5D0", "#F2E0C9", "#D0DDE8",
    "#E0D0E8", "#D5D0E8", "#EBDD9A"
]

# COLOUR_CYCLE = [
#     "#8E2F2F",  # arterial dark red
#     "#C96F6F",  # lighter arterial rose
#     "#2F6FB3",  # venous / RV blue
#     "#6F92C9",  # lighter venous blue
#     "#B78FA8",  # mauve chamber tone
#     "#D9A6A6",  # pale cardiac pink
#     "#8C7A7A",  # muted warm grey-taupe
#     "#A98FBF",  # soft lavender
# ]
#
# GROUP_BAND_COLOURS = [
#     "#EAD6D3",  # pale arterial blush
#     "#F1DFDC",  # soft rose
#     "#DCE6F2",  # pale blue
#     "#E7EDF7",  # lighter blue
#     "#E8DFE8",  # pale mauve
#     "#F2E6E6",  # pale pink
#     "#E6E1E1",  # warm light grey
#     "#E7E1EF",  # pale lavender
#     "#EEE8E4",  # soft neutral
# ]


def build_plot(output_params, param_counts, output_stem="edge_bundling"):
    """
    Build and save the hierarchical edge bundling plot.

    Parameters
    ----------
    output_params : dict    {output_name: {param: pct, ...}}
    param_counts  : Counter how many outputs each parameter appears in
    output_stem   : str     filename stem (without extension)
    """

    target_names_raw = list(output_params.keys())
    target_display = OrderedDict()
    target_colours = {}

    for i, raw_name in enumerate(target_names_raw):
        disp = TARGET_DISPLAY_NAMES.get(raw_name, raw_name)
        target_display[raw_name] = disp
        target_colours[raw_name] = COLOUR_CYCLE[i % len(COLOUR_CYCLE)]

    all_params_in_data = set()
    for params in output_params.values():
        all_params_in_data.update(params.keys())

    ordered_params = []
    active_groups = OrderedDict()
    accounted = set()

    for grp_name, members in GROUPS.items():
        active = [m for m in members if m in all_params_in_data]
        if active:
            active_groups[grp_name] = active
            ordered_params.extend(active)
            accounted.update(active)

    ungrouped = sorted(all_params_in_data - accounted)
    if ungrouped:
        active_groups["Other"] = ungrouped
        ordered_params.extend(ungrouped)

    n_params = len(ordered_params)
    n_groups = len(active_groups)

    GAP_DEG = 3.0
    ARC_SPAN = 360.0
    TOTAL_GAP = GAP_DEG * n_groups
    AVAILABLE_ARC = ARC_SPAN - TOTAL_GAP
    START_ANGLE = 90 + (360 - ARC_SPAN) / 2
    angle_per_param = AVAILABLE_ARC / n_params

    group_names_list = list(active_groups.keys())

    param_to_group = {}
    for grp_name, members in active_groups.items():
        for p in members:
            param_to_group[p] = grp_name

    def darken_hex(hex_colour, factor=0.75):
        hex_colour = hex_colour.lstrip("#")
        r = int(hex_colour[0:2], 16)
        g = int(hex_colour[2:4], 16)
        b = int(hex_colour[4:6], 16)

        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)

        return f"#{r:02X}{g:02X}{b:02X}"

    group_fill_colours = {}
    for gi, grp_name in enumerate(group_names_list):
        band_colour = GROUP_BAND_COLOURS[gi % len(GROUP_BAND_COLOURS)]
        group_fill_colours[grp_name] = darken_hex(band_colour, factor=0.9)

    group_start = {}
    idx = 0
    for grp_name, members in active_groups.items():
        group_start[grp_name] = idx
        idx += len(members)

    param_angles = {}
    current_angle = START_ANGLE
    current_group_idx = 0

    for i, p in enumerate(ordered_params):
        if (current_group_idx < n_groups - 1 and
                i == group_start[group_names_list[current_group_idx + 1]]):
            current_angle += GAP_DEG
            current_group_idx += 1
        param_angles[p] = current_angle
        current_angle += angle_per_param

    n_targets = len(target_names_raw)
    INNER_R = 0.30
    target_angles = {}
    for i, tname in enumerate(target_names_raw):
        target_angles[tname] = 360.0 * i / n_targets + 90

    OUTER_R = 1.0
    LABEL_R = 1.12

    fig, ax = plt.subplots(figsize=(14, 14), facecolor="white")
    ax.set_aspect("equal")
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.axis("off")

    # --- Group arc bands ---
    for gi, grp_name in enumerate(group_names_list):
        members = active_groups[grp_name]
        a_start = param_angles[members[0]] - angle_per_param * 0.5
        a_end = param_angles[members[-1]] + angle_per_param * 0.5

        thetas = np.linspace(np.radians(a_start), np.radians(a_end), 60)
        r_in, r_out = OUTER_R - 0.06, OUTER_R + 0.06

        xs_o = r_out * np.cos(thetas)
        ys_o = r_out * np.sin(thetas)
        xs_i = r_in * np.cos(thetas[::-1])
        ys_i = r_in * np.sin(thetas[::-1])
        verts = list(zip(xs_o, ys_o)) + list(zip(xs_i, ys_i))
        verts.append(verts[0])

        poly = plt.Polygon(
            verts,
            facecolor=GROUP_BAND_COLOURS[gi % len(GROUP_BAND_COLOURS)],
            edgecolor="none",
            alpha=0.45,
            zorder=0
        )
        ax.add_patch(poly)


        thetas = np.linspace(np.radians(a_start), np.radians(a_end), 60)
        OUTER_R1 = 1.5
        r_in, r_out = OUTER_R1 - 0.06, OUTER_R1 + 0.06
        xs_o = r_out * np.cos(thetas)
        ys_o = r_out * np.sin(thetas)
        xs_i = r_in * np.cos(thetas[::-1])
        ys_i = r_in * np.sin(thetas[::-1])
        verts = list(zip(xs_o, ys_o)) + list(zip(xs_i, ys_i))
        verts.append(verts[0])

        poly = plt.Polygon(
            verts,
            facecolor=GROUP_BAND_COLOURS[gi % len(GROUP_BAND_COLOURS)],
            edgecolor="none",
            alpha=0.45,
            zorder=0
        )
        ax.add_patch(poly)

        mid_a = np.radians((a_start + a_end) / 2)
        lx = (OUTER_R + 0.42) * np.cos(mid_a)
        ly = (OUTER_R + 0.42) * np.sin(mid_a)
        rot = np.degrees(mid_a)
        if 90 < rot % 360 < 270:
            rot += 180

        # ax.text(
        #     lx, ly, grp_name,
        #     ha="center", va="center",
        #     fontsize=11, fontweight="bold", color="#444444",
        #     rotation=rot, rotation_mode="anchor", fontstyle="italic"
        # )

    def cubic_bezier(p0, p1, p2, p3, n=80):
        t = np.linspace(0, 1, n)[:, None]
        return ((1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 +
                3 * (1 - t) * t**2 * p2 + t**3 * p3)

    # --- Edges ---
    max_sens = max(v for d in output_params.values() for v in d.values())

    OUTER_BUNDLE_R = 0.88
    INNER_BUNDLE_R = 0.88

    for raw_name, params in output_params.items():
        colour = target_colours[raw_name]
        ta = np.radians(target_angles[raw_name])
        tx, ty = INNER_R * np.cos(ta), INNER_R * np.sin(ta)

        for pname, pct in params.items():
            if pname not in param_angles:
                continue

            pa = np.radians(param_angles[pname])
            px, py = OUTER_R * np.cos(pa), OUTER_R * np.sin(pa)

            cp1 = np.array([
                INNER_BUNDLE_R * np.cos(ta),
                INNER_BUNDLE_R * np.sin(ta)
            ])

            cp2 = np.array([
                OUTER_BUNDLE_R * np.cos(pa),
                OUTER_BUNDLE_R * np.sin(pa)
            ])

            pts = cubic_bezier(
                np.array([tx, ty]),
                cp1,
                cp2,
                np.array([px, py]),
                n=80
            )

            alpha = 0.15 + 0.70 * (pct / max_sens)
            lw = 1.2 + 1.0 * (pct / max_sens)

            ax.plot(
                pts[:, 0], pts[:, 1],
                color=colour, alpha=alpha,
                linewidth=lw, solid_capstyle="round", zorder=1
            )

    # --- Parameter nodes ---
    for p in ordered_params:
        a = np.radians(param_angles[p])
        x, y = OUTER_R * np.cos(a), OUTER_R * np.sin(a)
        size = 15 + param_counts.get(p, 1) * 100

        grp_name = param_to_group[p]
        fill_colour = group_fill_colours[grp_name]

        ax.scatter(
            x, y,
            s=size,
            c=fill_colour,
            edgecolors="#555555",
            linewidths=1.2,
            zorder=5
        )

        deg = param_angles[p] % 360
        lx = LABEL_R * np.cos(a)
        ly = LABEL_R * np.sin(a)
        ha = "left" if (deg < 90 or deg > 270) else "right"
        rot = deg if (deg < 90 or deg > 270) else deg + 180

        label = PARAM_DISPLAY_NAMES.get(p, p.replace("_", " "))

        ax.text(
            lx, ly, label,
            ha=ha, va="center",
            fontsize=14, fontweight="medium", color="#333333",
            rotation=rot, rotation_mode="anchor",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
        )

    # --- Target nodes ---
    for raw_name in target_names_raw:
        ta = np.radians(target_angles[raw_name])
        tx, ty = INNER_R * np.cos(ta), INNER_R * np.sin(ta)
        colour = target_colours[raw_name]
        disp = target_display[raw_name]

        ax.scatter(
            tx, ty,
            s=350,
            c=colour,
            edgecolors="white",
            linewidths=2,
            zorder=10
        )
        ax.text(
            tx, ty - 0.09, disp,
            ha="center", va="top",
            fontsize=14, fontweight="bold", color=colour,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            zorder=11
        )

    # ── 5. LEGENDS ───────────────────────────────────────────────────────────

    legend_handles = []
    for raw_name in target_names_raw:
        label = target_display[raw_name].replace("\n", " ")
        legend_handles.append(
            mpatches.Patch(
                facecolor=target_colours[raw_name],
                edgecolor="none",
                label=label
            )
        )

    # # Size legend
    # size_ax = fig.add_axes([0.14, 0.10, 0.15, 0.10])
    # size_ax.set_xlim(0, 5)
    # size_ax.set_ylim(0, 3)
    # size_ax.axis("off")
    # size_ax.set_title(
    #     "Count (# targets sensitive)",
    #     fontweight="bold", fontsize=8.5, y=0.76, pad=0, loc="center"
    # )
    #
    # for i, cnt in enumerate([1, 3, 5]):
    #     s = 40 + cnt * 60
    #     size_ax.scatter(
    #         i * 1.6 + 0.8, 1.5,
    #         s=s,
    #         c="#B8B8B8",
    #         edgecolors="white",
    #         linewidths=1.2
    #     )
    #     size_ax.text(
    #         i * 1.6 + 0.8, 0.5, str(cnt),
    #         ha="center", va="center", fontsize=8, color="#555555", fontweight="bold",
    #     )
    #
    # # Opacity legend
    # op_ax = fig.add_axes([0.78, 0.12, 0.15, 0.08])
    # op_ax.set_xlim(0, 1)
    # op_ax.set_ylim(0, 1)
    # op_ax.axis("off")
    # op_ax.set_title(
    #     "Opacity = sensitivity %",
    #     fontsize=8.5, fontweight="bold", pad=0, loc="center", y=0.62
    # )
    #
    # grad = np.linspace(0, 1, 256).reshape(1, -1)
    # light_greys = colors.LinearSegmentedColormap.from_list(
    #     "light_greys",
    #     plt.cm.Greys(np.linspace(0.15, 0.75, 256))
    # )
    #
    # op_ax.imshow(
    #     grad,
    #     aspect="auto",
    #     cmap=light_greys,
    #     extent=(0.05, 0.95, 0.2, 0.5),
    #     alpha=0.9
    # )
    # op_ax.text(0.05, 0.0, "Low", fontsize=7, ha="center", va="top", fontweight="bold")
    # op_ax.text(0.95, 0.0, "High", fontsize=7, ha="center", va="top", fontweight="bold")
    #
    # fig.suptitle(
    #     "DGSM Sensitivity Analysis — Hierarchical Edge Bundling",
    #     fontsize=15, fontweight="bold", y=0.91, color="#222222"
    # )
    # fig.text(
    #     0.5, 0.88,
    #     "Parameters grouped by physiological subsystem · "
    #     "Node size ∝ # targets · Opacity ∝ DGSM %",
    #     ha="center", fontsize=9.5, color="#666666"
    # )

    plt.savefig(
        f"{output_stem}.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.3
    )
    plt.close()

    print(f"Parsed {len(output_params)} targets, "
          f"{len(ordered_params)} unique parameters "
          f"({n_groups} groups)")
    for raw_name, params in output_params.items():
        print(f"  {raw_name:25s}  ->  {len(params)} sensitive parameters")

# ── 6. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    filename = "C:/Users/vanes/Downloads/exercise_model/ODE_Exercise/Entire_system/DGSM_Rest_Paper/Resp_Groups.txt"
    output_stem = "Resp_edge_bundling"

    output_params, param_counts = parse_sensitivity_file(filename)
    build_plot(output_params, param_counts, output_stem=output_stem)
