"""
Plotting utilities for marine vessel simulations.
"""

import numpy as np


def add_force_direction_arrows(ax, direction_angle, spacing=2.0,
                                      color='0.2', alpha=0.25,
                                      coord_system='NED'):
    """
    Simplified background arrow tiling:
        tiles arrows with given spacing in data units,
        direction as unit vectors.

    :param ax: matplotlib.axes.Axes Axis to add arrows to.
    :param direction_angle: float Direction angle in radians.
        If coord_system='NED': 0 = north, pi/2 = east.
        If coord_system='math': 0 = east, pi/2 = north.
    :param spacing: float Grid spacing in data units (e.g., meters).
    :param color: str Color (grayscale string '0.0'~'1.0' or other matplotlib color).
    :param alpha: float Transparency (0-1).
    :param coord_system: str Coordinate system: 'NED' or 'math'.
    :return: matplotlib.quiver.Quiver Quiver object.
    :raises ValueError: If spacing <= 0.
    """

    # Get current axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    if spacing <= 0:
        raise ValueError("spacing must be > 0")

    # Generate grid points (include boundary to avoid missing edge columns)
    xs = np.arange(x_min, x_max + spacing, spacing)
    ys = np.arange(y_min, y_max + spacing, spacing)
    X, Y = np.meshgrid(xs, ys)

    # Convert direction to matplotlib math coordinates (0° = right, 90° = up)
    if coord_system == 'NED':
        # Convert NED to math: math_angle = pi/2 - ned_angle
        ang = np.pi / 2 - direction_angle
    else:
        ang = direction_angle

    # Unit vector components
    u = np.cos(ang)
    v = np.sin(ang)

    # Create arrays of same direction for all arrows
    U = np.full_like(X, u, dtype=float)
    V = np.full_like(Y, v, dtype=float)

    # Draw quiver using defaults (no custom scale/width/head* parameters)
    q = ax.quiver(X, Y, U, V, color=color, alpha=alpha)

    # Restore original axis limits (quiver might have changed them)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Prevent arrow from appearing in legend
    q.set_label('_nolegend_')

    return q

