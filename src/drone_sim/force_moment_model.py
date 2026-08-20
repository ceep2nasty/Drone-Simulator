import numpy as np


def calculate_body_forces_and_moments(thrusts: np.ndarray, reaction_torques: np.ndarray, motor_positions: np.ndarray, spin_directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the total body forces and moments acting on the drone, assuming that the thrust acts along the negative z-axis in the body frame.

    Parameters:
    - thrusts: A 1D array of thrust values for each motor. (4 for quadcopter)
    - reaction_torques: A 1D array of reaction torque values for each motor.
    - motor_positions: A 2D array where each row represents the (x, y, z) position of a motor relative to the drone's center of mass.
    - spin_directions: A 1D array indicating the spin direction of each motor, +1 for positive body-z reaction torque and -1 for negative body-z reaction torque

    Returns:
    - total_force: A 3D vector representing the total force acting on the drone in the body frame.
    - total_moment: A 3D vector representing the total moment acting on the drone in the body frame.
    """
    # Convert positive thrust magnitudes into forward-right-down body-frame vectors.

    motor_thrust = np.zeros_like(thrusts)
    for i in range(len(thrusts)):
        motor_thrust[i] = -thrusts[i]  # Thrust acts upward in body frame, opposite to the body z-axis (downward)

    # Total force in body frame (assuming thrust acts along the negative z-axis)
    total_force = np.array([0.0, 0.0, np.sum(motor_thrust)])

    # Total moment in body frame
    total_moment = np.zeros(3)
    for i in range(len(thrusts)):
        # Moment due to thrust
        moment_from_thrust = np.cross(motor_positions[i], np.array([0.0, 0.0, motor_thrust[i]]))
        total_moment += moment_from_thrust
        # Moment due to reaction torque
        moment_from_reaction = np.array([0.0, 0.0, spin_directions[i] * reaction_torques[i]])
        total_moment += moment_from_reaction

    return total_force, total_moment
