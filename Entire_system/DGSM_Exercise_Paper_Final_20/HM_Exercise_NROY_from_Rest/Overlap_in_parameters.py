# code for comparing between two DGSM runs for the overlap in sensitive parameters
import re
from collections import defaultdict, Counter

import torch


def parse_sensitivity_file(filename):
    """
    Parse sensitivity output file.
    Returns:
    - output_params: dict {output_name: set(parameters)}
    - param_counts: Counter of parameter appearances across outputs
    """
    output_params = defaultdict(set)
    current_output = None

    output_header = re.compile(r"^Output:\s*(.+)$")
    param_line = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*[-+0-9\.eE]+\s*\(")

    with open(filename, "r") as f:
        for line in f:
            header_match = output_header.search(line)
            if header_match:
                current_output = header_match.group(1).strip()
                continue

            param_match = param_line.match(line)
            if param_match and current_output is not None:
                param = param_match.group(1)
                output_params[current_output].add(param)

    param_counts = Counter()
    for params in output_params.values():
        param_counts.update(params)

    return output_params, param_counts


def get_all_params(output_params):
    """Flatten all parameters from all outputs into a single set."""
    all_params = set()
    for params in output_params.values():
        all_params.update(params)
    return all_params


def format_python_set(name, params):
    """Return a Python assignment for a sorted set of parameter names."""
    sorted_params = sorted(params, key=str.lower)
    if not sorted_params:
        return f"{name} = set()"

    values = ", ".join(repr(param) for param in sorted_params)
    return f"{name} = {{{values}}}"


if __name__ == "__main__":

    file20 = r"C:\Users\vanes\Downloads\exercise_model\ODE_Exercise\Entire_system\DGSM_Rest_Paper_Final_20\DGSM_20.txt"
    file50 = r"C:\Users\vanes\Downloads\exercise_model\ODE_Exercise\Entire_system\DGSM_Exercise_Paper_Final_20\DGSM_20_Exercise.txt"

    output_params_50, param_counts_50 = parse_sensitivity_file(file50)
    output_params_20, param_counts_20 = parse_sensitivity_file(file20)

    params_50 = get_all_params(output_params_50)
    params_20 = get_all_params(output_params_20)

    # ── Overlap analysis ──────────────────────────────────────────────────────
    overlap       = params_50 & params_20
    rest_only     = params_20 - params_50
    exercise_only = params_50 - params_20
    only_in_50    = exercise_only
    only_in_20    = rest_only

    print("=" * 80)
    print(f"DGSM_50  — unique parameters : {len(params_50)}")
    print(f"DGSM_20  — unique parameters : {len(params_20)}")
    print(f"Overlap (in both)            : {len(overlap)}")
    print(f"Only in DGSM_50              : {len(only_in_50)}")
    print(f"Only in DGSM_20              : {len(only_in_20)}")
    print("=" * 80)

    print(f"\n{'-'*40}")
    print(f"Parameters in BOTH files ({len(overlap)}):")
    print(f"{'-'*40}")
    for p in sorted(overlap, key=str.lower):
        print(f"  {p}")

    if only_in_50:
        print(f"\n{'-'*40}")
        print(f"Only in DGSM_50 ({len(only_in_50)}):")
        print(f"{'-'*40}")
        for p in sorted(only_in_50, key=str.lower):
            print(f"  {p}")

    if only_in_20:
        print(f"\n{'-'*40}")
        print(f"Only in DGSM_20 ({len(only_in_20)}):")
        print(f"{'-'*40}")
        for p in sorted(only_in_20, key=str.lower):
            print(f"  {p}")

    # ── Per-output overlap ────────────────────────────────────────────────────
    all_outputs = sorted(set(output_params_50) | set(output_params_20))

    print(f"\n{'='*100}")
    print("Per-output overlap")
    print(f"{'='*100}")
    print(f"{'Output':<30} {'50-only':>8} {'overlap':>8} {'20-only':>8}  overlapping parameters")
    print(f"{'-'*100}")

    for out in all_outputs:
        p50 = output_params_50.get(out, set())
        p20 = output_params_20.get(out, set())
        ov  = p50 & p20
        print(
            f"{out:<30} {len(p50-p20):>8} {len(ov):>8} {len(p20-p50):>8}"
            f"  {', '.join(sorted(ov, key=str.lower)) if ov else '-'}"
        )

    print()
    print("# Parameters in both files")
    print(format_python_set("subset_vars", overlap))
    print()
    print("# Only in rest/DGSM_20")
    print(format_python_set("subset_rest_only", rest_only))
    print()
    print("# Only in exercise/DGSM_50")
    print(format_python_set("subset_exercise_only", exercise_only))
