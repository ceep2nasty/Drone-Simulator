import numpy as np
import pytest

from drone_sim.translational_dynamics import calculate_translational_acceleration


# Test cases for the translational dynamics of the drone
def test_just_gravity() -> None:
    """Test the translational dynamics of the drone."""

    # Test case 1: no inertial force (just gravity acting on the drone)
    # Test 1 parameters
    force_inertial = np.array([0.0, 0.0, 0.0])
    mass = 1.0
    gravity = 9.81

    acceleration = calculate_translational_acceleration(force_inertial, mass, gravity)
    expected_acceleration = np.array([0.0, 0.0, -9.81])
    assert np.allclose(acceleration, expected_acceleration), (
        f"Expected {expected_acceleration}, but got {acceleration}"
    )


def test_just_horizontal_force() -> None:
    """Test the translational dynamics of the drone."""

    # Test case 2: horizontal force acting on the drone (no vertical force)
    # Test 2 parameters
    force_inertial = np.array([10, 0, 0])
    mass = 1.0
    gravity = 9.81

    acceleration = calculate_translational_acceleration(force_inertial, mass, gravity)
    expected_acceleration = np.array([10.0, 0.0, -9.81])
    assert np.allclose(acceleration, expected_acceleration), (
        f"Expected {expected_acceleration}, but got {acceleration}"
    )


def test_hover_force() -> None:
    """Test the translational dynamics of the drone."""

    # Test case 3: force equal to weight (hovering)
    # Test 3 parameters
    mass = 1.0
    gravity = 9.81
    force_inertial = np.array([0, 0, 9.81])  # Force equal to weight

    acceleration = calculate_translational_acceleration(force_inertial, mass, gravity)
    expected_acceleration = np.array([0, 0, 0])
    assert np.allclose(acceleration, expected_acceleration), (
        f"Expected {expected_acceleration}, but got {acceleration}"
    )


# Now test that the function raises errors for invalid inputs


def test_0_mass() -> None:
    """Test that the function raises an error for zero mass."""
    force_inertial = np.array([0.0, 0.0, 0.0])
    mass = 0.0
    gravity = 9.81

    with pytest.raises(ValueError, match="mass must be positive."):
        calculate_translational_acceleration(force_inertial, mass, gravity)


def test_force_inertial_invalid_shape() -> None:
    """Test that the function raises an error for invalid force_inertial shape."""
    force_inertial = np.array([0.0, 0.0])  # Invalid shape
    mass = 1.0
    gravity = 9.81

    with pytest.raises(ValueError, match="force_inertial must have shape \\(3,\\)."):
        calculate_translational_acceleration(force_inertial, mass, gravity)
