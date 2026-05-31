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
    filename = r"C:\Users\vanes\Downloads\exercise_model\ODE_Exercise\Entire_system\DGSM_Exercise_Paper_Final_20\DGSM_20_Exercise_Constant.txt"
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
