import pandas as pd
import re


def extract_biomarker_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    # Split into biomarker sections (biomarker name followed by a table)
    sections = re.split(r'\n\s*\n', text.strip())

    biomarker_data = {}
    biomarker_name = None

    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue

        # If a line ends with ":", it's the biomarker name
        if lines[0].endswith(":"):
            biomarker_name = lines[0].replace(":", "").strip()
            header = lines[1].split()
            data = []
            for line in lines[2:]:
                parts = line.split()
                if len(parts) == len(header) + 1:
                    param = parts[0]
                    values = parts[1:]
                    data.append([param] + values)
            df = pd.DataFrame(data, columns=["param"] + header)
            biomarker_data[biomarker_name] = df

    return biomarker_data


def find_overlap(biomarker_data):
    # Find intersection of parameter names across all biomarkers
    sets = [set(df['param']) for df in biomarker_data.values()]
    overlap = set.intersection(*sets)
    return sorted(list(overlap))


# Example usage
file_path = "C:/Users/vanes/Desktop/Heart Rate.txt"
biomarker_data = extract_biomarker_data(file_path)
overlap_params = find_overlap(biomarker_data)

print("Overlap parameters across all biomarkers:")
print(overlap_params)

# Optionally, save the overlap table for inspection
overlap_dfs = {name: df[df['param'].isin(overlap_params)] for name, df in biomarker_data.items()}
with pd.ExcelWriter("biomarker_overlap.xlsx") as writer:
    for name, df in overlap_dfs.items():
        df.to_excel(writer, sheet_name=name, index=False)
