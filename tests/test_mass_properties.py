import numpy as np
import pytest

from drone_sim.mass_properties import (
    RigidComponent,
    calculate_center_of_mass,
    calculate_combined_rotation,
    calculate_inertia_tensor,
    calculate_rotation_matrix,
    rotate_inertia_tensor,
)


def test_rigid_component_initialization():
    position = np.array([1.0, 2.0, 3.0])
    rotation = np.eye(3)
    component = RigidComponent(
        name="test_component",
        shape="rectangular_prism",
        mass=5.0,
        position=position,
        dimensions=(1.0, 2.0, 3.0),
        rotation=rotation,
    )
    assert component.name == "test_component"
    assert component.shape == "rectangular_prism"
    assert component.mass == 5.0
    assert np.array_equal(component.position, position)
    assert component.dimensions == (1.0, 2.0, 3.0)
    assert np.array_equal(component.rotation, rotation)


def test_rectangular_prism_inertia():
    component = RigidComponent(
        name="test_component",
        shape="rectangular_prism",
        mass=5.0,
        position=np.array([0.0, 0.0, 0.0]),
        dimensions=(1.0, 2.0, 3.0),
    )
    inertia = component.local_inertia()
    expected_inertia = np.diag(
        [
            (1 / 12) * 5.0 * (2.0**2 + 3.0**2),
            (1 / 12) * 5.0 * (1.0**2 + 3.0**2),
            (1 / 12) * 5.0 * (1.0**2 + 2.0**2),
        ]
    )
    assert np.array_equal(inertia, expected_inertia)


def test_solid_cylinder_inertia():
    component = RigidComponent(
        name="test_component",
        shape="solid_cylinder",
        mass=5.0,
        position=np.array([0.0, 0.0, 0.0]),
        dimensions=(1.0, 2.0),
    )
    inertia = component.local_inertia()
    expected_inertia = np.diag(
        [
            (1 / 12) * 5.0 * (3 * 1.0**2 + 2.0**2),
            (1 / 12) * 5.0 * (3 * 1.0**2 + 2.0**2),
            (1 / 2) * 5.0 * 1.0**2,
        ]
    )
    assert np.array_equal(inertia, expected_inertia)


def test_thin_disk_inertia():
    component = RigidComponent(
        name="test_component",
        shape="thin_disk",
        mass=5.0,
        position=np.array([0.0, 0.0, 0.0]),
        dimensions=(1.0,),
    )
    inertia = component.local_inertia()
    expected_inertia = np.diag(
        [(1 / 4) * 5.0 * 1.0**2, (1 / 4) * 5.0 * 1.0**2, (1 / 2) * 5.0 * 1.0**2]
    )
    assert np.array_equal(inertia, expected_inertia)


def test_default_rotation_matrix():
    component = RigidComponent(
        name="test_component",
        shape="rectangular_prism",
        mass=5.0,
        position=np.array([0.0, 0.0, 0.0]),
        dimensions=(1.0, 2.0, 3.0),
    )
    assert np.array_equal(component.rotation, np.eye(3))


def test_invalid_mass_raises_error() -> None:
    with pytest.raises(ValueError, match="Mass must be positive"):
        RigidComponent(
            name="invalid",
            shape="rectangular_prism",
            mass=0.0,
            position=np.zeros(3),
            dimensions=(1.0, 1.0, 1.0),
        )


def test_rotation_about_z_swaps_xy() -> None:
    rotation_z_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    component = RigidComponent(
        name="test_component",
        shape="rectangular_prism",
        mass=5.0,
        position=np.array([0.0, 0.0, 0.0]),
        dimensions=(1.0, 2.0, 3.0),
        rotation=rotation_z_90,
    )
    assert np.array_equal(component.rotation, rotation_z_90)
    Ixx, Iyy, _Izz = component.local_inertia().diagonal()
    inertia_fusion = component.inertia_in_fusion_frame()
    inertia_fusion_diag = inertia_fusion.diagonal()
    assert np.isclose(inertia_fusion_diag[0], Iyy)
    assert np.isclose(inertia_fusion_diag[1], Ixx)


def test_easy_calculate_center_of_mass():
    components = [
        RigidComponent(
            name="component1",
            shape="rectangular_prism",
            mass=1.0,
            dimensions=(1.0, 1.0, 1.0),
            position=np.array([1.0, 0.0, 0.0]),
        ),
        RigidComponent(
            name="component2",
            shape="rectangular_prism",
            mass=1.0,
            dimensions=(1.0, 1.0, 1.0),
            position=np.array([-1.0, 0.0, 0.0]),
        ),
    ]
    center_of_mass = calculate_center_of_mass(components)
    expected_center_of_mass = np.array([0.0, 0.0, 0.0])
    assert np.allclose(center_of_mass, expected_center_of_mass)


def test_unequal_calculate_center_of_mass():
    components = [
        RigidComponent(
            name="component1",
            shape="rectangular_prism",
            mass=2.0,
            dimensions=(1.0, 1.0, 1.0),
            position=np.array([1.0, 0.0, 0.0]),
        ),
        RigidComponent(
            name="component2",
            shape="rectangular_prism",
            mass=1.0,
            dimensions=(1.0, 1.0, 1.0),
            position=np.array([-1.0, 0.0, 0.0]),
        ),
    ]
    center_of_mass = calculate_center_of_mass(components)
    expected_center_of_mass = np.array([1 / 3, 0.0, 0.0])
    assert np.allclose(center_of_mass, expected_center_of_mass)


def test_calculate_total_inertia() -> None:
    components = [
        RigidComponent(
            name="component1",
            shape="rectangular_prism",
            mass=2.0,
            dimensions=(1.0, 1.0, 1.0),
            position=np.array([1.0, 0.0, 0.0]),
        ),
        RigidComponent(
            name="component2",
            shape="thin_disk",
            mass=1.0,
            dimensions=(1.0,),
            position=np.array([-1.0, 0.0, 0.0]),
        ),
    ]

    total_inertia = calculate_inertia_tensor(components)

    prism_inertia = (1 / 12) * 2.0 * (1.0**2 + 1.0**2)

    disk_ixx = (1 / 4) * 1.0 * 1.0**2
    disk_iyy = disk_ixx
    disk_izz = (1 / 2) * 1.0 * 1.0**2

    parallel_axis_contribution = 2.0 * (2.0 / 3.0) ** 2 + 1.0 * (4.0 / 3.0) ** 2

    expected_inertia = np.diag(
        [
            prism_inertia + disk_ixx,
            prism_inertia + disk_iyy + parallel_axis_contribution,
            prism_inertia + disk_izz + parallel_axis_contribution,
        ]
    )

    np.testing.assert_allclose(total_inertia, expected_inertia)


def test_rotation_about_z() -> None:
    rotation = calculate_rotation_matrix(90.0, "z")
    vector = np.array([1.0, 0.0, 0.0])

    rotated_vector = rotation @ vector

    expected = np.array([0.0, 1.0, 0.0])

    np.testing.assert_allclose(
        rotated_vector,
        expected,
        atol=1e-12,
    )


def test_combined_rotation() -> None:
    rotation_parameters_1 = (90.0, "z")
    rotation_parameters_2 = (90.0, "y")

    combined_rotation = calculate_combined_rotation(
        [
            rotation_parameters_1,
            rotation_parameters_2,
        ]
    )

    rotation_matrix_1 = calculate_rotation_matrix(*rotation_parameters_1)
    rotation_matrix_2 = calculate_rotation_matrix(*rotation_parameters_2)

    expected_rotation = rotation_matrix_2 @ rotation_matrix_1

    np.testing.assert_allclose(
        combined_rotation,
        expected_rotation,
        atol=1e-12,
    )


def test_rotated_matrix() -> None:
    original_matrix = np.diag([1, 2, 3])
    rotation_parameters_1 = (90.0, "z")
    rotation_parameters_2 = (90.0, "y")
    combined_rotation_matrix = calculate_combined_rotation(
        [
            rotation_parameters_1,
            rotation_parameters_2,
        ]
    )
    rotated_matrix = (
        combined_rotation_matrix @ original_matrix @ combined_rotation_matrix.T
    )

    expected = np.diag([3, 1, 2])
    np.testing.assert_allclose(
        rotated_matrix,
        expected,
        atol=1e-12,
    )


def test_rotate_inertia_tensor_about_z() -> None:
    inertia_tensor = np.diag([1.0, 2.0, 3.0])

    rotations = [
        (90.0, "z"),
    ]

    actual = rotate_inertia_tensor(
        inertia_tensor,
        rotations,
    )

    expected = np.diag([2.0, 1.0, 3.0])

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-12,
    )
