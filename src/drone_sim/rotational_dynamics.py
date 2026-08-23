import numpy as np


def calculate_angular_acceleration(
    moment_body: np.ndarray,
    angular_velocity_body: np.ndarray,
    inertia_body: np.ndarray,
) -> np.ndarray:
    """Calculate angular acceleration in the body frame."""

    # Confirm input types and shapes
    if not isinstance(moment_body, np.ndarray):
        raise TypeError("moment_body must be a NumPy array")
    if moment_body.shape != (3,):
        raise ValueError("moment_body must have shape (3,)")

    if not isinstance(angular_velocity_body, np.ndarray):
        raise TypeError("angular_velocity_body must be a NumPy array")
    if angular_velocity_body.shape != (3,):
        raise ValueError("angular_velocity_body must have shape (3,)")

    if not isinstance(inertia_body, np.ndarray):
        raise TypeError("inertia_body must be a NumPy array")
    if inertia_body.shape != (3, 3):
        raise ValueError("inertia_body must have shape (3, 3)")

    # Confirm inertia tensor is symmetric and positive definite
    if not np.allclose(inertia_body, inertia_body.T):
        raise ValueError("inertia_body must be a symmetric matrix")

    try:
        np.linalg.cholesky(inertia_body)
    except np.linalg.LinAlgError as error:
        raise ValueError("inertia_body must be positive definite") from error

    angular_momentum = inertia_body @ angular_velocity_body

    gyroscopic_moment = np.cross(
        angular_velocity_body,
        angular_momentum,
    )

    net_dynamic_moment = moment_body - gyroscopic_moment

    angular_acceleration_body = np.linalg.solve(
        inertia_body,
        net_dynamic_moment,
    )

    return angular_acceleration_body
