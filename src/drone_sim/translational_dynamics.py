from numbers import Real

import numpy as np


def calculate_translational_acceleration(
    force_inertial: np.ndarray,
    mass: Real,
    gravity: Real,
) -> np.ndarray:
    """Calculate translational acceleration in the inertial frame."""

    if not isinstance(force_inertial, np.ndarray):
        raise TypeError("force_inertial must be a NumPy array.")
    if force_inertial.shape != (3,):
        raise ValueError("force_inertial must have shape (3,).")
    if not isinstance(mass, Real):
        raise TypeError("mass must be a real number.")
    if not isinstance(gravity, Real):
        raise TypeError("gravity must be a real number.")
    if mass <= 0.0:
        raise ValueError("mass must be positive.")
    if gravity < 0.0:
        raise ValueError("gravity must be non-negative.")

    gravity_inertial = np.array([0.0, 0.0, -gravity])
    return force_inertial / mass + gravity_inertial
