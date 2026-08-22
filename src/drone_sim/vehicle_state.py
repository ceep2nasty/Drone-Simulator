from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleState:
    """Represents the state of the drone
    Attributes:
        position (np.ndarray): The position of the drone in 3D space (x, y, z).
        velocity (np.ndarray): The velocity of the drone in 3D space (vx, vy, vz).
        orientation (np.ndarray): The orientation of the drone represented as euler angles (roll, pitch, yaw).
        angular_velocity (np.ndarray): The angular velocity of the drone in 3D space (wx, wy, wz).
    """

    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self) -> None:
        # Ensure that all attributes are numpy arrays
        self.position = np.asarray(self.position, dtype=float).copy()
        self.velocity = np.asarray(self.velocity, dtype=float).copy()
        self.orientation = np.asarray(self.orientation, dtype=float).copy()
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=float).copy()

        # Validate the shapes of the arrays
        if self.position.shape != (3,):
            raise ValueError("Position must be a 3D vector.")
        if self.velocity.shape != (3,):
            raise ValueError("Velocity must be a 3D vector.")
        if self.orientation.shape != (3,):
            raise ValueError("Orientation must be a 3D vector.")
        if self.angular_velocity.shape != (3,):
            raise ValueError("Angular velocity must be a 3D vector.")
