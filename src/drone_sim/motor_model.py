import numpy as np

def calculate_motor_outputs(
    commands: np.ndarray,
    omega_max: float,
    thrust_coefficient: float,
    torque_coefficient: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert normalized motor commands to thrust and torque outputs.
    """
    # Ensure commands are within [0, 1]
    clipped_commands = np.clip(commands, 0.0, 1.0)

    # Calculate the angular velocity for each motor
    angular_speeds = clipped_commands * omega_max
    thrusts = thrust_coefficient * angular_speeds**2
    torques = torque_coefficient * angular_speeds**2

    return angular_speeds,thrusts, torques