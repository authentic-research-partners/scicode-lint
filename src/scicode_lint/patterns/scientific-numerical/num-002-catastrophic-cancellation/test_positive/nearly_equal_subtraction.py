import numpy as np


def exponential_difference(x):
    val1 = np.exp(x)
    val2 = np.exp(x - 1e-8)
    diff = val1 - val2
    return diff


def beam_deflection_change(load_kn, length_m, elasticity, inertia):
    """Compute change in beam deflection after a small load increase."""
    base_load = load_kn
    updated_load = load_kn + 0.001
    deflection_before = (base_load * length_m**3) / (48 * elasticity * inertia)
    deflection_after = (updated_load * length_m**3) / (48 * elasticity * inertia)
    delta = deflection_after - deflection_before
    return delta


def distance_from_origin(point1, point2):
    dist1 = np.sqrt(point1[0] ** 2 + point1[1] ** 2 + point1[2] ** 2)
    dist2 = np.sqrt(point2[0] ** 2 + point2[1] ** 2 + point2[2] ** 2)
    diff = dist1 - dist2
    return diff


large_val = 1e15
small_delta = 1.0
result = (large_val + small_delta) - large_val
