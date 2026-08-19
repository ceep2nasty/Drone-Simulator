import numpy as np

from drone_sim.force_moment_model import calculate_body_forces_and_moments


# Test basic functionality of calculate_body_forces_and_moments
def test_calculate_body_forces_and_moments() -> None:
    # Test parameters
    thrusts = np.array([2.0, 2.0, 2.0, 2.0])  # Thrusts for each motor
    reaction_torques = np.array([0.1, 0.1, 0.1, 0.1])  # Reaction torques for each motor
    motor_positions = np.array(
        [[1, -1, 0], [1, 1, 0], [-1, -1, 0], [-1, 1, 0]]
    )  # Positions of motors in body frame
    spin_directions = np.array([1, -1, -1, 1])

    # Calculate forces and moments
    total_force, total_moment = calculate_body_forces_and_moments(
        thrusts, reaction_torques, motor_positions, spin_directions
    )

    # Expected results
    expected_total_force = np.array([0.0, 0.0, -8.0])  # Total thrust in body frame
    expected_total_moment = np.array([0.0, 0.0, 0.0])  # Total moment in body frame

    # Assertions
    np.testing.assert_allclose(total_force, expected_total_force)
    np.testing.assert_allclose(total_moment, expected_total_moment)


# Test with thrust on just one motor


def test_single_motor_thrust() -> None:
    # Test parameters
    thrusts = np.array([2.0, 0.0, 0.0, 0.0])  # Thrusts for each motor
    reaction_torques = np.array(
        [
            0.1,
            0.0,
            0.0,
            0.0,
        ]
    )  # Reaction torques for each motor
    motor_positions = np.array(
        [[1, -1, 0], [1, 1, 0], [-1, -1, 0], [-1, 1, 0]]
    )  # Positions of motors in body frame
    spin_directions = np.array([1, -1, -1, 1])

    # Calculate forces and moments
    total_force, total_moment = calculate_body_forces_and_moments(
        thrusts, reaction_torques, motor_positions, spin_directions
    )
    # Expected results
    expected_total_force = np.array([0.0, 0.0, -2.0])  # Total thrust in body frame
    expected_total_moment = np.array([2.0, 2.0, 0.1])  # Total moment in body frame]

    # Assertions
    np.testing.assert_allclose(total_force, expected_total_force)
    np.testing.assert_allclose(total_moment, expected_total_moment)
