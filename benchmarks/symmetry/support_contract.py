"""Independent fixed-pencil contracts, including FFT residuals from stored fields.

Run after sampling. NumPy/SciPy are verification tools, never part of the timed
solver. Native certificates and acceptance flags are checked, not trusted alone.
"""
import math
import numpy as np
from scipy.fft import fft2, ifft2


def decode(bits):
    return np.asarray(bits, dtype=np.uint64).view(np.float64)


def fields(document):
    data = decode(document["field_bits"])
    if data.ndim != 3 or data.shape[2] != 2 or not np.isfinite(data).all():
        raise ValueError("invalid complete complex field payload")
    return data[:, :, 0] + 1j * data[:, :, 1]


def material(numerics):
    identity = numerics["operator_identity"]
    nx, ny = numerics["grid"]
    size = nx * ny
    values = decode(identity["operator_bits"])
    tensor_flag = identity["operator_bits"][9 + 2 * size]
    if (identity["nx"] != nx or identity["ny"] != ny or identity["storage_bits"] != 64
            or identity["centered"] or tensor_flag not in (0,1)
            or len(values) != 10 + (6 if tensor_flag else 2) * size):
        raise ValueError("unsupported or malformed operator identity")
    eps, inv = values[9:9+size], values[9+size:9+2*size]
    if not np.isfinite(values).all() or np.min(eps) <= 0 or np.min(inv) <= 0:
        raise ValueError("nonfinite/nonpositive scalar material")
    if identity["operator_bits"][2] != int(numerics["polarization"] == "TE"):
        raise ValueError("polarization and exact operator identity disagree")
    tensor = values[10+2*size:].reshape(ny,nx,2,2) if tensor_flag else None
    if tensor is not None and (np.max(np.abs(tensor[:,:,0,1]-tensor[:,:,1,0]))>1e-12
                               or np.min(np.linalg.eigvalsh(tensor))<=0):
        raise ValueError("inverse material tensor is not symmetric positive definite")
    return values, eps, inv, tensor


def independent_certificate(numerics, document, projected=True):
    """Apply the sampled scalar/tensor pencil with independent unitary FFTs."""
    nx, ny = numerics["grid"]
    raw, eps, inv, tensor = material(numerics)
    u = fields(document)
    values = np.asarray(document["spectrum"], dtype=float)
    if u.shape != (len(values), nx * ny) or not np.isfinite(values).all():
        raise ValueError("spectrum and physical field shapes disagree")
    f = fft2(u.reshape(-1, ny, nx), norm="ortho", workers=1)
    # Blaze's original FFT rectangle stores the even Nyquist as positive.
    gx = np.arange(nx); gx = np.where(gx <= nx // 2, gx, gx - nx)
    gy = np.arange(ny); gy = np.where(gy <= ny // 2, gy, gy - ny)
    kx = raw[3] + raw[5] * gx[None, :] + raw[7] * gy[:, None]
    ky = raw[4] + raw[6] * gx[None, :] + raw[8] * gy[:, None]
    if numerics["polarization"] == "TM":
        a = (kx*kx + ky*ky) * f
        b = fft2((u * eps).reshape(-1, ny, nx), norm="ortho", workers=1)
    else:
        gradients = [ifft2(k*f,norm="ortho",workers=1) for k in (kx,ky)]
        flux = [inv.reshape(ny,nx)*v for v in gradients] if tensor is None else [
            sum(tensor[:,:,i,j]*gradients[j] for j in range(2)) for i in range(2)]
        a = sum(k*fft2(v,norm="ortho",workers=1) for k,v in zip((kx,ky),flux))
        b = f.copy()
    leakage = 0.0
    if projected:
        modes = np.asarray(numerics["support_identity"]["modes"], dtype=int)
        iy, ix = modes[:, 1] % ny, modes[:, 0] % nx
        mask = np.ones((ny, nx), dtype=bool); mask[iy, ix] = False
        leakage = float(np.max(np.linalg.norm(f[:, mask], axis=1) / np.linalg.norm(u, axis=1)))
        a, b, coordinates = a[:, iy, ix], b[:, iy, ix], f[:, iy, ix]
    else:
        a, b, coordinates = a.reshape(len(u), -1), b.reshape(len(u), -1), f.reshape(len(u), -1)
    numerator = np.linalg.norm(a - values[:, None]*b, axis=1)
    denominator = np.linalg.norm(a, axis=1) + np.abs(values)*np.linalg.norm(b, axis=1)
    residuals = np.divide(numerator, denominator, out=numerator.copy(), where=denominator>0)
    absolute = numerator / np.linalg.norm(coordinates, axis=1)
    gram = coordinates.conj() @ b.T
    return dict(residuals=residuals.tolist(), absolute_residuals=absolute.tolist(),
                b_orthogonality_defect=float(np.max(np.abs(gram-np.eye(len(u))))),
                max_support_leakage=leakage)


def certificate_issues(certificate, width, kernel, label):
    problems = []
    for key in ("residuals", "absolute_residuals"):
        vals = certificate.get(key, [])
        if (len(vals) != width or any(type(x) not in (int, float) or not math.isfinite(x) or x<0 for x in vals)):
            return [label + " has malformed " + key]
    orth = certificate.get("b_orthogonality_defect")
    if type(orth) not in (float, int) or not math.isfinite(orth) or not 0 <= orth <= 1e-8:
        problems.append(label + " fails B orthogonality")
    if any(x > 1e-5 for x in certificate["residuals"][int(kernel):]):
        problems.append(label + " fails projected residual")
    if kernel and certificate["absolute_residuals"][0] > 1e-7:
        problems.append(label + " fails absolute kernel residual")
    return problems


def quality(n):
    problems, independent = [], None
    try:
        q, result, space = n["bands"], n["result"], n["support_identity"]
        if n.get("error") is not None or not isinstance(result, dict):
            return dict(issues=["solve rejected: " + str(n.get("error"))])
        spectrum, report = result["spectrum"], result["report"]
        k = decode(space["k_bits"])
        kernel = any(np.array_equal(k + np.asarray(g), [0.0, 0.0]) for g in space["modes"])
        if (len(spectrum) != q or result["field_count"] != q or len(result["field_bits"]) != q
                or result["field_payload_bytes"] != q * math.prod(n["grid"]) * 16
                or not all(type(x) in (float, int) and math.isfinite(x) and x>=-1e-9 for x in spectrum)
                or any(b < a-1e-9 for a, b in zip(spectrum, spectrum[1:]))):
            problems.append("invalid complete ordered spectrum/output")
        if result["support_identity"] != space or space["grid"] != n["grid"]:
            problems.append("inconsistent support identity")
        if report["accepted"] is not True or report["solver_converged"] is not True or report["rejection"] is not None:
            problems.append("solver rejected or failed convergence")
        if report["coefficient_count"] != len(space["modes"]) or report["analytic_kernel"] != kernel:
            problems.append("finite dimension or kernel ownership mismatch")
        problems += certificate_issues(report["projected"], q, kernel, "reported")
        independent = independent_certificate(n, result)
        problems += certificate_issues(independent, q, kernel, "independent FFT")
        if independent["max_support_leakage"] > 1e-10:
            problems.append("physical output lies outside retained support")
        sectors = n["sectors"]
        if n["mode"] == "unreduced":
            if sectors is not None or report["method"] != "lobpcg":
                problems.append("unreduced policy changed")
        else:
            d, attempts, qualification = sectors["descriptors"], sectors["attempts"], sectors["qualification"]
            if not sectors["accepted"] or sectors["rejection"] is not None or not qualification["authorized"]:
                problems.append("sector or qualification rejection")
            if (sectors["finite_dimension"] != len(space["modes"])
                    or sectors["sum_sector_dimensions"] != len(space["modes"])
                    or sum(x["rank"]*x["partner_rows"] for x in d) != len(space["modes"])
                    or sectors["kernel_owners"] != int(kernel)
                    or sum(x["owns_analytic_kernel"] for x in d) != int(kernel)):
                problems.append("incomplete dimension/multiplicity/kernel accounting")
            if not attempts or sum(x["iterations"] for x in attempts) != report["iterations"]:
                problems.append("attempted iterations lost")
            if any(not a["accepted"] or not a["solver_converged"] or a["rejection"] is not None for a in attempts):
                problems.append("failed attempt in accepted run")
            latest = {}
            for a in attempts:
                descriptor = d[a["sector"]]
                width = a["prefix"] * descriptor["partner_rows"]
                problems += certificate_issues(a["projected"], width, descriptor["owns_analytic_kernel"], "sector attempt")
                if (not math.isfinite(a["max_leakage"]) or not 0 <= a["max_leakage"] <= 1e-8
                        or a["reconstructed_partners"] != a["prefix"]*(descriptor["partner_rows"]-1)
                        or not 0 < a["prefix"] <= descriptor["rank"]
                        or (a["dense_complete"] and a["iterations"] != 0)):
                    problems.append("invalid sector attempt / reconstruction")
                latest[a["sector"]] = a
            if sum(a["prefix"]*d[i]["partner_rows"] for i,a in latest.items()) != sectors["computed_with_multiplicity"]:
                problems.append("retained multiplicity differs from final attempts")
            labels = sectors["output_labels"]
            if len(labels) != q or len({(x["sector"],x["row"],x["local_index"]) for x in labels}) != q:
                problems.append("missing/duplicated output labels")
            tol = qualification["tolerance"]
            if qualification["rejection"] is not None or qualification["maximum_multiplier_defect"] > tol:
                problems.append("qualification multiplier failure")
            for action in qualification["actions"]:
                if (not action["sample_permutation_exact"] or not action["support_closed"] or action["probe_count"]<=0
                        or any(not math.isfinite(action[key]) or not 0<=action[key]<=tol for key in
                               ("metric_defect","material_defect","tensor_defect","operator_defect","mass_defect"))):
                    problems.append("qualification structural/numerical failure")
    except (KeyError, TypeError, ValueError, IndexError, np.linalg.LinAlgError) as e:
        problems.append("malformed numerical record: " + str(e))
    return dict(issues=problems, independent_certificate=independent)


def common_guarded_prefix(a, b, limit):
    prefix = min(limit, len(a)-1, len(b)-1)
    while prefix > 0 and any(v[prefix]-v[prefix-1] <= 1e-6*max(1.0, abs(v[prefix])) for v in (a,b)):
        prefix -= 1
    return prefix


def compare(numerics, a, b):
    """Compare ordered frequencies and complete physical B-metric subspaces."""
    av, bv = np.asarray(a["spectrum"]), np.asarray(b["spectrum"])
    count = min(len(av),len(bv))
    errors = np.abs(np.sqrt(np.maximum(av[:count],0))-np.sqrt(np.maximum(bv[:count],0))) / np.maximum(1,np.sqrt(np.maximum(bv[:count],0)))
    prefix = common_guarded_prefix(av,bv,count-1)
    sine = None
    if prefix > 0:
        _, eps, _, _ = material(numerics)
        weight = eps if numerics["polarization"] == "TM" else np.ones_like(eps)
        left, right = fields(a)[:prefix], fields(b)[:prefix]
        overlap = left.conj() @ (right*weight).T
        squared = np.linalg.eigvalsh(overlap.conj().T @ overlap)
        sine = float(np.sqrt(1-np.clip(squared[0],0,1)))
    return dict(max_scaled_frequency_error=float(np.max(errors)),
                scaled_frequency_errors=errors.tolist(), guarded_prefix=prefix, max_sin_angle=sine)


def pair_quality(a,b):
    problems=[]
    if any(a[key] != b[key] for key in ("case","grid","bands","polarization","radius","precision","eigensolver","operator_identity","support_identity")):
        problems.append("paired physical pencil/configuration differs")
    comparison = None
    try:
        comparison = compare(a,a["result"],b["result"])
        if comparison["max_scaled_frequency_error"] > 1e-6:
            problems.append("paired frequencies disagree")
        if comparison["guarded_prefix"] <= 0 or comparison["max_sin_angle"] > 1e-3:
            problems.append("paired guarded physical subspaces disagree")
    except (KeyError, TypeError, ValueError, IndexError, np.linalg.LinAlgError) as e:
        problems.append("pair comparison unavailable: " + str(e))
    return dict(issues=problems, comparison=comparison)
