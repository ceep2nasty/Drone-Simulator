from numbers import Real

import numpy as np


def update_translational_state(
        position_inertial: np.ndarray,
        velocity_inertial: np.ndarray,
        acceleration_inertial: np.ndarray,
        dt: Real
) -> tuple[np.ndarray, np.ndarray]:
    """Update the translational state of the drone.

    Args:
        position_inertial (np.ndarray): The current position of the drone in the inertial frame.
        velocity_inertial (np.ndarray): The current velocity of the drone in the inertial frame.
        acceleration_inertial (np.ndarray): The current acceleration of the drone in the inertial frame.
        dt (Real): The time step for the update.

    Returns:
        tuple[np.ndarray, np.ndarray]: The updated position and velocity of the drone.
    """

    # Validate inputs
    if not isinstance(dt, Real):
        raise TypeError("dt must be a real number.")
    if not isinstance(position_inertial, np.ndarray):
        raise TypeError("position_inertial must be a NumPy array.")
    if position_inertial.shape != (3,):
        raise ValueError("position_inertial must have shape (3,).")
    if not isinstance(velocity_inertial, np.ndarray):
        raise TypeError("velocity_inertial must be a NumPy array.")
    if velocity_inertial.shape != (3,):
        raise ValueError("velocity_inertial must have shape (3,).")
    if not isinstance(acceleration_inertial, np.ndarray):
        raise TypeError("acceleration_inertial must be a NumPy array.")
    if acceleration_inertial.shape != (3,):
        raise ValueError("acceleration_inertial must have shape (3,).")
    
    new_position = position_inertial + velocity_inertial * dt + 0.5 * acceleration_inertial * dt ** 2 # semi-implicit Euler integration assuming constant acceleration over the time step
    new_velocity = velocity_inertial + acceleration_inertial * dt

    return [new_position, new_velocity]