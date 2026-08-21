import numpy as np


def as_cell_array(value, nxyz, name):
    """Return scalar or array input as a 1D array with one value per model cell."""
    if value is None:
        raise ValueError(f"{name} must be provided when DDMT is enabled")

    arr = np.asarray(value, dtype=float)

    if arr.ndim == 0:
        return np.full(nxyz, float(arr), dtype=float)

    arr = arr.reshape(-1)
    if arr.size != nxyz:
        raise ValueError(f"{name} must be scalar or length {nxyz}; got length {arr.size}")

    return arr


def validate_ddmt_arrays(theta_mobile, theta_immobile, alpham, nxyz):
    """Validate and normalize DDMT parameter arrays."""
    theta_mobile = as_cell_array(theta_mobile, nxyz, "theta_mobile")
    theta_immobile = as_cell_array(theta_immobile, nxyz, "theta_immobile")
    alpham = as_cell_array(alpham, nxyz, "alpham")

    if np.any(theta_mobile < 0.0):
        raise ValueError("theta_mobile must be nonnegative")
    if np.any(theta_immobile < 0.0):
        raise ValueError("theta_immobile must be nonnegative")
    if np.any(alpham < 0.0):
        raise ValueError("alpham must be nonnegative")

    active = (theta_mobile > 0.0) & (theta_immobile > 0.0) & (alpham > 0.0)
    if not np.any(active):
        raise ValueError("DDMT is enabled, but no cells have active mobile/immobile exchange")

    return theta_mobile, theta_immobile, alpham


def exchange_first_order_single_rate(mobile_conc_m3, immobile_conc_m3, theta_mobile, theta_immobile, alpham, dt):
    """Apply exact first-order single-rate mobile/immobile mass exchange."""
    cm = np.asarray(mobile_conc_m3, dtype=float)
    cim = np.asarray(immobile_conc_m3, dtype=float)

    if cm.shape != cim.shape:
        raise ValueError(
            f"mobile and immobile concentration arrays must have same shape; "
            f"got {cm.shape} and {cim.shape}"
        )

    ncomps, nxyz = cm.shape

    theta_m = np.asarray(theta_mobile, dtype=float).reshape(nxyz)
    theta_i = np.asarray(theta_immobile, dtype=float).reshape(nxyz)
    alpha = np.asarray(alpham, dtype=float).reshape(nxyz)

    cm_new = cm.copy()
    cim_new = cim.copy()

    active = (theta_m > 0.0) & (theta_i > 0.0) & (alpha > 0.0)
    if not np.any(active):
        return cm_new, cim_new

    tm = theta_m[active].reshape(1, -1)
    ti = theta_i[active].reshape(1, -1)
    zz = alpha[active].reshape(1, -1)

    total = tm * cm[:, active] + ti * cim[:, active]
    diff = cm[:, active] - cim[:, active]
    lam = zz * (1.0 / tm + 1.0 / ti)
    diff_new = diff * np.exp(-lam * dt)

    denom = tm + ti
    cm_new[:, active] = (total + ti * diff_new) / denom
    cim_new[:, active] = (total - tm * diff_new) / denom

    return cm_new, cim_new