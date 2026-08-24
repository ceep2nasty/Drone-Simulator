import numpy as np


def euler_rate_transform(
        omega: np.ndarray,
        eta: np.ndarray,
        dt: float,
        tol: float = 1e-12,
) -> np.ndarray:
    """Applies euler transformation matrix T(phi, theta) to state attitude (p, q, r) to obtain 
    euler rates. Then, integrates forward to obtain the updated angle rates.
    """
    phi = eta[0]
    theta = eta[1]
    
    # detect gimbal lock condition
    if abs(np.cos(theta)) < tol:
        raise ValueError("Gimbal lock condition detected")

    T = np.array([
    [1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
    [0.0, np.cos(phi), -np.sin(phi)],
    [0.0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
    ])
    p, q, r = omega[0], omega[1], omega[2]

    eta_dot = T @ np.array([p, q, r])
    return eta + eta_dot * dt

