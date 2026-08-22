import numpy as np
import pytest

from drone_sim.vehicle_state import VehicleState


def test_vehicle_state_initialization() -> None:
    # Test data
    position = np.array([1.0, 2.0, 3.0])
    velocity = np.array([0.5, 0.5, 0.5])
    orientation = np.array([0.1, 0.2, 0.3])
    angular_velocity = np.array([0.01, 0.02, 0.03])

    # Create VehicleState instance
    vehicle_state = VehicleState(
        position=position,
        velocity=velocity,
        orientation=orientation,
        angular_velocity=angular_velocity,
    )

    # Assertions
    assert np.allclose(vehicle_state.position, position)
    assert np.allclose(vehicle_state.velocity, velocity)
    assert np.allclose(vehicle_state.orientation, orientation)
    assert np.allclose(vehicle_state.angular_velocity, angular_velocity)


def test_vehicle_state_type_conversion() -> None:
    # Test data with lists instead of numpy arrays and ensures datatype converts to float
    position = [1.0, 2.0, 3.0]
    velocity = [0.5, 0.5, 0.5]
    orientation = [0.1, 0.2, 0.3]
    angular_velocity = [0.01, 0.02, 0.03]

    # Create VehicleState instance
    vehicle_state = VehicleState(
        position=position,
        velocity=velocity,
        orientation=orientation,
        angular_velocity=angular_velocity,
    )

    # Assertions
    assert np.allclose(vehicle_state.position, position)
    assert np.allclose(vehicle_state.velocity, velocity)
    assert np.allclose(vehicle_state.orientation, orientation)
    assert np.allclose(vehicle_state.angular_velocity, angular_velocity)

    assert np.issubdtype(vehicle_state.position.dtype, np.floating)
    assert np.issubdtype(vehicle_state.velocity.dtype, np.floating)
    assert np.issubdtype(vehicle_state.orientation.dtype, np.floating)
    assert np.issubdtype(vehicle_state.angular_velocity.dtype, np.floating)


def test_vehicle_state_invalid_shapes() -> None:
    # Test data with invalid shapes
    invalid_position = np.array(
        [1.0, 2.0]
    )  # Invalid shape, can also check by making it valid and ensuring error is raised
    velocity = np.array([0.5, 0.5, 0.5])
    orientation = np.array([0.1, 0.2, 0.3])
    angular_velocity = np.array([0.01, 0.02, 0.03])

    with pytest.raises(ValueError, match="Position must be a 3D vector."):
        VehicleState(
            position=invalid_position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
        )


def test_vehicle_state_defensive_copy() -> None:
    # Test data
    position = np.array([1.0, 2.0, 3.0])
    velocity = np.array([0.5, 0.5, 0.5])
    orientation = np.array([0.1, 0.2, 0.3])
    angular_velocity = np.array([0.01, 0.02, 0.03])

    # Create VehicleState instance
    vehicle_state = VehicleState(
        position=position,
        velocity=velocity,
        orientation=orientation,
        angular_velocity=angular_velocity,
    )

    # Modify the original arrays
    position[0] = 10.0
    velocity[1] = 10.0
    orientation[2] = 10.0
    angular_velocity[0] = 10.0

    # Assertions to ensure that the VehicleState instance has not been affected by changes to the original arrays
    assert not np.allclose(vehicle_state.position, position)
    assert not np.allclose(vehicle_state.velocity, velocity)
    assert not np.allclose(vehicle_state.orientation, orientation)
    assert not np.allclose(vehicle_state.angular_velocity, angular_velocity)
