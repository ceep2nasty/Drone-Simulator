import numpy as np

from drone_sim.motor_model import calculate_motor_outputs

def test_calculate_motor_outputs() -> None:
    # Test parameters
    commands = np.array([0.0, 0.25, 0.5, 1.0])
    omega_max = 1000.0
    thrust_coefficient = 1e-6
    torque_coefficient = 1e-8

    angular_speeds, thrusts, torques = calculate_motor_outputs(
        commands, omega_max, thrust_coefficient, torque_coefficient)

    # Expected results
    expected_angular_speeds = commands * omega_max
    expected_thrusts = thrust_coefficient * expected_angular_speeds**2
    expected_torques = torque_coefficient * expected_angular_speeds**2

    # Assertions
    np.testing.assert_allclose(angular_speeds, expected_angular_speeds)
    np.testing.assert_allclose(thrusts, expected_thrusts)
    np.testing.assert_allclose(torques, expected_torques)

def test_motor_commands_are_clipped() -> None:
    # Test parameters
    commands = np.array([-0.5, 0.0, 1.0, 1.5])
    omega_max = 1000.0
    thrust_coefficient = 1e-6
    torque_coefficient = 1e-8

    angular_speeds, thrusts, torques = calculate_motor_outputs(
        commands, omega_max, thrust_coefficient, torque_coefficient)

    # Expected Results
    expected_angular_speeds = np.clip(commands, 0.0, 1.0) * omega_max
    np.testing.assert_allclose(angular_speeds, expected_angular_speeds)

