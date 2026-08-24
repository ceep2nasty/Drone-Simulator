import numpy as np
import pytest

from drone_sim.update_angular_velocity import update_angular_velocity_func

# Test 1: ensure type error works as expected (use angular acceleration)


def test_type_error() -> None:
    # test parameters
    angular_velocity_body = np.array([0.0, 0.0, 0.0])
    dt = 1.0
    angular_acceleration_body = [1.0, 1.0, 1.0]

    with pytest.raises(TypeError):
        update_angular_velocity_func(
            angular_velocity_body, dt, angular_acceleration_body
        )


# Test 2: zero acceleration


def test_zero_accel() -> None:
    # test parameters
    angular_velocity_body = np.array([1.0, 1.0, 1.0])
    dt = 1.0
    angular_acceleration_body = np.array([0.0, 0.0, 0.0])

    calculated_update = update_angular_velocity_func(
        angular_acceleration_body=angular_acceleration_body,
        dt=dt,
        angular_velocity_body=angular_velocity_body,
    )

    expected = angular_velocity_body
    assert np.allclose(calculated_update, expected)


# Test 3: test to ensure that update creates a new vector and doesn't mutate predefined states
def test_update_does_not_mutate_inputs_or_return_same_array() -> None:
    angular_velocity_body = np.array([1.0, -2.0, 0.5])
    angular_acceleration_body = np.array([2.0, 4.0, -1.0])
    dt = 0.5

    original_omega = angular_velocity_body.copy()
    original_alpha = angular_acceleration_body.copy()

    new_omega = update_angular_velocity_func(
        angular_velocity_body=angular_velocity_body,
        dt=dt,
        angular_acceleration_body=angular_acceleration_body,
    )

    np.testing.assert_array_equal(angular_velocity_body, original_omega)
    np.testing.assert_array_equal(angular_acceleration_body, original_alpha)
    assert new_omega is not angular_velocity_body
