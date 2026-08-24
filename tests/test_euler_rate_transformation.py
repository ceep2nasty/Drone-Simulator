import numpy as np
import pytest

from drone_sim.euler_rate_transformation import euler_rate_transform


def test_level_attitude_matches_body_rates() -> None:
    omega = np.array([1.0, 2.0, 3.0])
    theta = 0.0
    phi = 0.0
    eta = np.array([phi, theta, 0.0])
    dt = 0.1

    updated_eta = euler_rate_transform(
        omega = omega,
        eta=eta,
        dt=dt,
    )

    expected_eta = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(updated_eta, expected_eta)

def test_zero_angular_velocity() -> None:
    omega = np.zeros(3)
    theta = 0.4
    phi = 0.7
    eta = np.array([phi, theta, 1.2])
    dt = 0.1

    updated_eta = euler_rate_transform(omega = omega, eta = eta, dt = dt)
    np.testing.assert_allclose(updated_eta, eta)

def test_roll_angle_coupling() -> None:
    omega = np.array([0.0, 1.0, 0.0])
    theta = 0.0
    phi = np.pi / 2
    eta = np.array([phi, theta, 0.0])
    dt = 1.0   

    updated_eta = euler_rate_transform(omega = omega, eta = eta, dt = dt)

    expected_eta = np.array([np.pi / 2, 0.0, 1.0])
    np.testing.assert_allclose(updated_eta, expected_eta, atol = 1e-12)

def test_body_yaw_rate_at_nonzero_pitch() -> None:
    omega = np.array([0.0, 0.0, 1.0])
    theta = np.pi / 6
    phi = 0.0
    eta = np.array([phi, theta, 0.0])
    dt = 1.0

    updated_eta = euler_rate_transform(omega = omega, eta = eta, dt = dt)

    expected_eta = np.array(
        [
            np.tan(np.pi / 6),
            theta,
            1.0 / np.cos(np.pi / 6),
        ]
    )
    np.testing.assert_allclose(updated_eta, expected_eta, atol = 1e-12)

def test_gimbal_lock_is_rejected() -> None:
    omega = np.ones(3)
    theta = np.pi / 2
    phi = 0.0
    eta = np.array([phi, theta, 0.])
    dt = 0.1

    with pytest.raises(ValueError, match="Gimbal lock"):
        euler_rate_transform(omega = omega, eta= eta, dt = dt)

def test_orientation_is_not_mutated() -> None:
    omega = np.array([1.0, 2.0, 3.0])
    eta = np.array([0.2, -0.1, 0.4])
    original_eta = eta.copy()
    dt = 0.1

    updated_eta = euler_rate_transform(omega = omega, eta = eta, dt = dt)

    np.testing.assert_array_equal(eta, original_eta)
    assert updated_eta is not eta

def test_integration_over_multiple_timesteps() -> None:
    omega = np.array([1.0, 0.0, 0.0])
    eta = np.array([0.0, 0.0, 0.0])
    dt = 0.1

    expected_orientations = [
    np.array([0.1, 0.0, 0.0]),
    np.array([0.2, 0.0, 0.0]),
    np.array([0.3, 0.0, 0.0]),
    np.array([0.4, 0.0, 0.0]),
    np.array([0.5, 0.0, 0.0]),
    ]

    for expected_eta in expected_orientations:
        eta = euler_rate_transform(omega = omega, eta = eta, dt = dt)
        np.testing.assert_allclose(eta, expected_eta, atol = 1e-12)