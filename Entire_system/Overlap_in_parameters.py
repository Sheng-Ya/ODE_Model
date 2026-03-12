import re
from collections import defaultdict, Counter
from pathlib import Path

def parse_sensitivity_file(filename):
    """
    Parse sensitivity output file.

    Returns:
    - output_params: dict {output_name: set(parameters)}
    - param_counts: Counter of parameter appearances across outputs
    """
    output_params = defaultdict(set)
    current_output = None

    # Detect output headers
    output_header = re.compile(r"^Output:\s*(.+)$")

    # Detect parameter lines
    param_line = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:")

    with open(filename, "r") as f:
        for line in f:
            # Check for new output section
            header_match = output_header.search(line)
            if header_match:
                current_output = header_match.group(1).strip()
                continue

            # Extract parameter names
            param_match = param_line.match(line)
            if param_match and current_output is not None:
                param = param_match.group(1)
                output_params[current_output].add(param)

    # Count how many outputs each parameter appears in
    param_counts = Counter()
    for params in output_params.values():
        param_counts.update(params)

    return output_params, param_counts


if __name__ == "__main__":
    filename = "C:/Users/vanes/Downloads/exercise_model/ODE_Exercise/Entire_system/DGSM_bounds/DGSM_20_rest.txt"
    output_params, param_counts = parse_sensitivity_file(filename)

    # ---- Global union ----
    all_params = sorted(param_counts.keys())
    print("\n" + "=" * 80)
    print(f"Total unique parameters: {len(all_params)}")
    print("=" * 80)
    for p in all_params:
        print(p)

    # # ---- Counts ----
    # print("\n" + "=" * 80)
    # print("Parameter appearance counts (across outputs)")
    # print("=" * 80)
    # for p, c in param_counts.most_common():
    #     print(f"{p:25s} : {c}")

    # Invert mapping: parameter -> outputs
    param_to_outputs = defaultdict(list)
    for output, params in output_params.items():
        for p in params:
            param_to_outputs[p].append(output)

    # Sort by number of outputs influenced (descending), then name
    sorted_params = sorted(
        param_to_outputs.items(),
        key=lambda x: (-len(x[1]), x[0])
    )

    print("=" * 100)
    print("Parameter influence summary (sorted by number of outputs)")
    print("=" * 100)

    for param, outputs in sorted_params:
        outputs_sorted = sorted(outputs)
        print(
            f"{param:25s} : "
            f"{len(outputs_sorted)} outputs -> "
            f"{', '.join(outputs_sorted)}"
        )

    txt = Path(filename).read_text().splitlines()

    params = set()
    for line in txt:
        m = re.match(r'^\s*([A-Za-z0-9_]+)\s*:\s*[-+0-9\.eE]+\s*\(', line)
        if m:
            params.add(m.group(1))

    params = sorted(params, key=str.lower)
    print(len(params))
    print(params)


# import re
# from collections import defaultdict, Counter
#
# def parse_sensitivity_file(filename):
#     """
#     Parse sensitivity output file.
#     Returns:
#     - output_params: dict {output_name: set(parameters)}
#     - param_counts: Counter of parameter appearances across outputs
#     """
#     output_params = defaultdict(set)
#     current_output = None
#
#     output_header = re.compile(r"^Output:\s*(.+)$")
#     param_line = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*[-+0-9\.eE]+\s*\(")
#
#     with open(filename, "r") as f:
#         for line in f:
#             header_match = output_header.search(line)
#             if header_match:
#                 current_output = header_match.group(1).strip()
#                 continue
#
#             param_match = param_line.match(line)
#             if param_match and current_output is not None:
#                 param = param_match.group(1)
#                 output_params[current_output].add(param)
#
#     param_counts = Counter()
#     for params in output_params.values():
#         param_counts.update(params)
#
#     return output_params, param_counts
#
#
# def get_all_params(output_params):
#     """Flatten all parameters from all outputs into a single set."""
#     all_params = set()
#     for params in output_params.values():
#         all_params.update(params)
#     return all_params
#
#
# if __name__ == "__main__":
#     file50 = "C:/Users/vanes/Downloads/exercise_model/ODE_Exercise/Entire_system/DGSM_bounds/DGSM_50_rest.txt"
#     file20 = "C:/Users/vanes/Downloads/exercise_model/ODE_Exercise/Entire_system/DGSM_bounds/DGSM_20_rest.txt"
#
#     output_params_50, param_counts_50 = parse_sensitivity_file(file50)
#     output_params_20, param_counts_20 = parse_sensitivity_file(file20)
#
#     params_50 = get_all_params(output_params_50)
#     params_20 = get_all_params(output_params_20)
#
#     # ── Overlap analysis ──────────────────────────────────────────────────────
#     overlap      = params_50 & params_20
#     only_in_50   = params_50 - params_20
#     only_in_20   = params_20 - params_50
#
#     print("=" * 80)
#     print(f"DGSM_50  — unique parameters : {len(params_50)}")
#     print(f"DGSM_20  — unique parameters : {len(params_20)}")
#     print(f"Overlap (in both)            : {len(overlap)}")
#     print(f"Only in DGSM_50              : {len(only_in_50)}")
#     print(f"Only in DGSM_20              : {len(only_in_20)}")
#     print("=" * 80)
#
#     print(f"\n{'─'*40}")
#     print(f"Parameters in BOTH files ({len(overlap)}):")
#     print(f"{'─'*40}")
#     for p in sorted(overlap, key=str.lower):
#         print(f"  {p}")
#
#     if only_in_50:
#         print(f"\n{'─'*40}")
#         print(f"Only in DGSM_50 ({len(only_in_50)}):")
#         print(f"{'─'*40}")
#         for p in sorted(only_in_50, key=str.lower):
#             print(f"  {p}")
#
#     if only_in_20:
#         print(f"\n{'─'*40}")
#         print(f"Only in DGSM_20 ({len(only_in_20)}):")
#         print(f"{'─'*40}")
#         for p in sorted(only_in_20, key=str.lower):
#             print(f"  {p}")
#
#     # ── Per-output overlap ────────────────────────────────────────────────────
#     all_outputs = sorted(set(output_params_50) | set(output_params_20))
#
#     print(f"\n{'='*100}")
#     print("Per-output overlap")
#     print(f"{'='*100}")
#     print(f"{'Output':<30} {'50-only':>8} {'overlap':>8} {'20-only':>8}  overlapping parameters")
#     print(f"{'─'*100}")
#
#     for out in all_outputs:
#         p50 = output_params_50.get(out, set())
#         p20 = output_params_20.get(out, set())
#         ov  = p50 & p20
#         print(
#             f"{out:<30} {len(p50-p20):>8} {len(ov):>8} {len(p20-p50):>8}"
#             f"  {', '.join(sorted(ov, key=str.lower)) if ov else '—'}"
#         )