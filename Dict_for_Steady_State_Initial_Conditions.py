import numpy as np
from Initial_Conditions_after_running_again import Initial_Conditions

state_names = list(Initial_Conditions.keys())
combined = np.load("combined.npy")

def _fmt_num(x: float, zero_tol: float = 1e-9) -> str:
    x = float(x)

    # clamp tiny values to exactly 0.0
    if abs(x) < zero_tol:
        return "0.0"

    # scientific notation for very small/very large (but not clamped)
    if abs(x) < 1e-3 or abs(x) >= 1e6:
        return f"{x:.8e}".replace("e-0", "e-").replace("e+0", "e+")

    # otherwise compact decimal
    return f"{x:.9g}"

def format_initial_conditions(keys, values,
                              cardio_marker="theta_change_O2_sp",
                              gas_marker="Pd_1_O2",
                              indent=4,
                              force_zero_keys=("VE_integral",),
                              zero_tol=1e-9) -> str:
    keys = list(keys)
    values = np.asarray(values).ravel()
    assert len(keys) == len(values)

    force_zero_keys = set(force_zero_keys)

    def _fmt_num(x: float) -> str:
        x = float(x)
        if abs(x) < zero_tol:
            return "0.0"
        if abs(x) < 1e-3 or abs(x) >= 1e6:
            return f"{x:.8e}".replace("e-0", "e-").replace("e+0", "e+")
        return f"{x:.9g}"

    i_cardio = keys.index(cardio_marker) if cardio_marker in keys else None
    i_gas    = keys.index(gas_marker)    if gas_marker in keys    else None

    lines = ["Initial_Conditions = {"]

    def emit_block(k0, k1, header=None):
        if header is not None:
            lines.append("")
            lines.append(f"{' '*indent}# {header}")
        for k, v in zip(keys[k0:k1], values[k0:k1]):
            if k in force_zero_keys:
                v = 0.0
            lines.append(f"{' '*indent}'{k}': {_fmt_num(v)},")

    if i_cardio is None or i_gas is None or not (0 <= i_cardio <= i_gas <= len(keys)):
        emit_block(0, len(keys))
    else:
        emit_block(0, i_cardio)
        emit_block(i_cardio, i_gas, header="Cardio controller")
        emit_block(i_gas, len(keys), header="Gas exchange")

    lines.append("")
    lines.append("}")
    return "\n".join(lines)


print(format_initial_conditions(state_names, combined))