import numpy as np


def update_angular_velocity_func(
    angular_velocity_body: np.ndarray,
    angular_acceleration_body: np.ndarray,
    dt: float,
) -> np.ndarray:
    if not isinstance(angular_acceleration_body, np.ndarray):
        raise TypeError(
            f"Expected np.ndarray for angular acceleration, got {type(angular_acceleration_body).__name__}"
        )
    if angular_acceleration_body.shape != (3,):
        raise ValueError(
            f"Expected angular acceleration shape (3,), got {angular_acceleration_body.shape}"
        )
    return angular_velocity_body + angular_acceleration_body * dt
