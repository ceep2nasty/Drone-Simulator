import numpy as np

from drone_sim.rotational_dynamics import calculate_angular_acceleration


# Test 1 (0?): test rotational equilibrium case
def test_rotational_equilibrium() -> None:
    # Test parameters
    moment_body = np.array(
        [0.0, 0.0, 0.0],
    )
    angular_velocity_body = np.array(
        [0.0, 0.0, 0.0],
    )
    inertia_body = np.eye(3, 3)

    angular_acceleration = calculate_angular_acceleration(
        moment_body, angular_velocity_body, inertia_body
    )
    expected = np.array([0.0, 0.0, 0.0])

    assert np.allclose(angular_acceleration, expected)


# Test 2: test single axis moment from rest
def test_single_axis_moment() -> None:
    # Test parameters
    moment_body = np.array(
        [6.0, 0.0, 0.0],
    )
    angular_velocity_body = np.array(
        [0.0, 0.0, 0.0],
    )
    inertia_body = np.diag([2, 3, 4])

    expected = np.array([3.0, 0.0, 0.0])
    angular_acceleration = calculate_angular_acceleration(
        moment_body, angular_velocity_body, inertia_body
    )
    assert np.allclose(angular_acceleration, expected)


# Test 3: gyroscopic coupling with no applied moment
def test_gyro_couple() -> None:
    # Test parameters
    moment_body = np.array(
        [0.0, 0.0, 0.0],
    )
    angular_velocity_body = np.array(
        [1.0, 2.0, 3.0],
    )
    inertia_body = np.diag([2, 3, 4])

    expected = np.array([-3.0, 2.0, -0.5])
    angular_acceleration = calculate_angular_acceleration(
        moment_body, angular_velocity_body, inertia_body
    )
    assert np.allclose(angular_acceleration, expected)
