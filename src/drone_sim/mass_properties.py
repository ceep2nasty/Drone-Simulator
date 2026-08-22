from dataclasses import dataclass, field

import numpy as np


@dataclass
class RigidComponent:
    name: str
    shape: str
    mass: float
    position: np.ndarray
    dimensions: tuple[float, ...]
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.rotation = np.asarray(self.rotation, dtype=float)

        if self.mass <= 0.0:
            raise ValueError("Mass must be positive.")

        if self.position.shape != (3,):
            raise ValueError("Position must have shape (3,).")

        if self.rotation.shape != (3, 3):
            raise ValueError("Rotation matrix must have shape (3, 3).")

    def local_inertia(self) -> np.ndarray:
        """Calculate the local inertia tensor based on the shape and dimensions."""
        if self.shape == "rectangular_prism":
            lx, ly, lz = self.dimensions
            Ixx = (1 / 12) * self.mass * (ly**2 + lz**2)
            Iyy = (1 / 12) * self.mass * (lx**2 + lz**2)
            Izz = (1 / 12) * self.mass * (lx**2 + ly**2)
            return np.diag([Ixx, Iyy, Izz])
        elif self.shape == "solid_cylinder":
            radius, height = self.dimensions
            Ixx = (1 / 12) * self.mass * (3 * radius**2 + height**2)
            Iyy = Ixx
            Izz = (1 / 2) * self.mass * radius**2
            return np.diag([Ixx, Iyy, Izz])
        elif self.shape == "thin_disk":
            (radius,) = self.dimensions
            Ixx = (1 / 4) * self.mass * radius**2
            Iyy = Ixx
            Izz = (1 / 2) * self.mass * radius**2
            return np.diag([Ixx, Iyy, Izz])
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")

    def inertia_in_fusion_frame(self) -> np.ndarray:
        """Calculate the inertia tensor in the fusion frame using the rotation matrix."""
        local_inertia = self.local_inertia()
        return self.rotation @ local_inertia @ self.rotation.T


def calculate_total_mass(components: list[RigidComponent]) -> float:
    """Calculate the total mass of a list of components."""
    return sum(component.mass for component in components)


def calculate_center_of_mass(components: list[RigidComponent]) -> np.ndarray:
    """Calculate the center of mass of a list of components."""
    total_mass = calculate_total_mass(components)
    if total_mass == 0:
        raise ValueError("Total mass is zero, cannot compute center of mass.")
    weighted_positions = sum(
        component.mass * component.position for component in components
    )
    return weighted_positions / total_mass


def calculate_inertia_tensor(components: list[RigidComponent]) -> np.ndarray:
    """Calculate the total inertia tensor of a list of components in the fusion frame."""
    inertia_tensor = np.zeros((3, 3))
    center_of_mass = calculate_center_of_mass(components)
    # Apply parallel axis theorem to shift each component's inertia tensor to the center of mass
    for component in components:
        r = component.position - center_of_mass
        inertia_tensor += component.inertia_in_fusion_frame() + component.mass * (
            np.dot(r, r) * np.eye(3) - np.outer(r, r)
        )
    return inertia_tensor


def calculate_rotation_matrix(
    theta: float,
    axis: str = "no_rotation",
) -> np.ndarray:
    """Return a right-handed rotation matrix for theta degrees."""
    theta_rad = np.deg2rad(theta)

    if axis == "x":
        rotation_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta_rad), -np.sin(theta_rad)],
                [0.0, np.sin(theta_rad), np.cos(theta_rad)],
            ]
        )

    elif axis == "y":
        rotation_matrix = np.array(
            [
                [np.cos(theta_rad), 0.0, np.sin(theta_rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta_rad), 0.0, np.cos(theta_rad)],
            ]
        )

    elif axis == "z":
        rotation_matrix = np.array(
            [
                [np.cos(theta_rad), -np.sin(theta_rad), 0.0],
                [np.sin(theta_rad), np.cos(theta_rad), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    elif axis == "no_rotation":
        rotation_matrix = np.eye(3)

    else:
        raise ValueError("Axis must be 'x', 'y', 'z', or 'no_rotation'.")

    return rotation_matrix


def calculate_combined_rotation(rotations: list[tuple[float, str]]) -> np.ndarray:
    """Combine rotations in the listed order"""
    combined_rotation = np.eye(3, 3)
    for theta, axis in rotations:
        rotation = calculate_rotation_matrix(theta, axis)
        combined_rotation = rotation @ combined_rotation

    return combined_rotation


def rotate_inertia_tensor(
    inertia_tensor: np.ndarray,
    rotations: list[tuple[float, str]],
) -> np.ndarray:
    """Rotate an inertia tensor using the specified rotations."""
    rotation_matrix = calculate_combined_rotation(rotations)

    return rotation_matrix @ inertia_tensor @ rotation_matrix.T
