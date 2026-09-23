"""Additional gates for matched native and WASM interface experiments."""
import copy
import json


def physical_configuration(result):
    """Return the complete comparison contract, allowing only declared policies.

    Return None for missing/malformed configuration or mismatched output layout.
    Backend identifiers and timing metadata are deliberately outside this key;
    geometry, sampling, solver controls and output requirements are inside it.
    """
    config = result.get("configuration")
    arrays = result.get("arrays")
    if not isinstance(config, dict) or not isinstance(arrays, dict):
        return None
    try:
        json.dumps(config, allow_nan=False)
        grid = result["grid"]
        bands = result["bands"]
        count = result["expected_points"]
        fields = result["field_output"]
        if (not isinstance(grid, list) or len(grid) != 2
                or any(type(n) is not int or n <= 0 for n in grid)
                or type(bands) is not int or bands <= 0
                or type(count) is not int or count <= 0 or type(fields) is not bool):
            return None
        if (config["grid"]["resolution"] != grid
                or config["bands"]["count"] != bands
                or config["polarization"] != result["polarization"]
                or config["eigensolver"]["precision"] != result["precision"]
                or config["eigensolver"]["tolerance"] != result["tolerance"]
                or config["results"]["eigenvectors"] is not fields):
            return None
        expected = {
            "frequencies": ("float64", [count, bands], count * bands * 8),
            "k_points": ("float64", [count, 2], count * 2 * 8),
            "residuals": ("float64", [count, bands], count * bands * 8),
            "absolute_residuals": ("float64", [count, bands], count * bands * 8),
        }
        if fields:
            expected["eigenvectors"] = (
                "complex128", [count, bands, grid[1], grid[0]],
                count * bands * grid[0] * grid[1] * 16)
        elif "eigenvectors" in arrays:
            return None
        for name, (dtype, shape, size) in expected.items():
            a = arrays[name]
            if (a["dtype"] != dtype or a["shape"] != shape
                    or type(a["payload_bytes"]) is not int or a["payload_bytes"] != size):
                return None
        normalized = copy.deepcopy(config)
        # Both controls must still have explicit policies. Requested-path gates
        # separately verify that the solver actually executed each policy.
        symmetry = normalized["bands"]["symmetry"]
        del symmetry["k_reuse"]
        del symmetry["arithmetic"]
        return normalized
    except (KeyError, TypeError, ValueError):
        return None


def same_interface_jobs(a, b):
    ca, cb = physical_configuration(a), physical_configuration(b)
    if ca is None or ca != cb:
        return False
    # Isospectral stars can hide a wrong output permutation if only sorted
    # frequencies are compared. Keep the reported requested point order too.
    try:
        ka, kb = [[p["k"] for p in r["points"]] for r in (a, b)]
        return len(ka) == a["expected_points"] and ka == kb
    except (KeyError, TypeError):
        return False
