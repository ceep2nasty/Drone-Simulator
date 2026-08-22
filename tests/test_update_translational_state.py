import numpy as np

from drone_sim.update_translational_state import update_translational_state
from drone_sim.vehicle_state import VehicleState


def test_update_translational_state_zero_acceleration() -> None:
    """Test the update_translational_state function."""

    # Test case 1: simple case with zero acceleration
    position_inertial = np.array([0.0, 0.0, 0.0])
    velocity_inertial = np.array([1.0, 0.0, 0.0])
    acceleration_inertial = np.array([0.0, 0.0, 0.0])
    dt = 1.0

    new_position, new_velocity = update_translational_state(
        position_inertial, velocity_inertial, acceleration_inertial, dt
    )

    expected_position = np.array([1.0, 0.0, 0.0])
    expected_velocity = np.array([1.0, 0.0, 0.0])

    assert np.allclose(new_position, expected_position), (
        f"Expected {expected_position}, but got {new_position}"
    )
    assert np.allclose(new_velocity, expected_velocity), (
        f"Expected {expected_velocity}, but got {new_velocity}"
    )


def test_update_translational_state_constant_acceleration() -> None:
    """Test the update_translational_state function with constant acceleration."""

    # Test case 2: simple case with constant acceleration
    position_inertial = np.array([0.0, 0.0, 0.0])
    velocity_inertial = np.array([1.0, 0.0, 0.0])
    acceleration_inertial = np.array([1.0, 0.0, 0.0])
    dt = 1.0

    new_position, new_velocity = update_translational_state(
        position_inertial, velocity_inertial, acceleration_inertial, dt
    )

    expected_position = np.array([1.5, 0.0, 0.0])  # s = ut + 1/2 at^2
    expected_velocity = np.array([2.0, 0.0, 0.0])  # v = u + at

    assert np.allclose(new_position, expected_position), (
        f"Expected {expected_position}, but got {new_position}"
    )
    assert np.allclose(new_velocity, expected_velocity), (
        f"Expected {expected_velocity}, but got {new_velocity}"
    )


def test_update_translational_state_to_vehicle() -> None:
    """Test the update_translational_state function and convert to VehicleState."""

    # Test case 3: simple case with constant acceleration
    position_inertial = np.array([0.0, 0.0, 0.0])
    velocity_inertial = np.array([1.0, 0.0, 0.0])
    acceleration_inertial = np.array([1.0, 0.0, 0.0])
    dt = 1.0

    # parameters for vehicle state
    orientation = np.array([0.0, 0.0, 0.0])
    angular_velocity = np.array([0.0, 0.0, 0.0])

    new_position, new_velocity = update_translational_state(
        position_inertial, velocity_inertial, acceleration_inertial, dt
    )

    vehicle_state = VehicleState(
        position=new_position,
        velocity=new_velocity,
        orientation=orientation,
        angular_velocity=angular_velocity,
    )

    expected_position = np.array([1.5, 0.0, 0.0])  # s = ut + 1/2 at^2
    expected_velocity = np.array([2.0, 0.0, 0.0])  # v = u + at

    assert np.allclose(vehicle_state.position, expected_position), (
        f"Expected {expected_position}, but got {vehicle_state.position}"
    )
    assert np.allclose(vehicle_state.velocity, expected_velocity), (
        f"Expected {expected_velocity}, but got {vehicle_state.velocity}"
    )
