"""
Plotting utilities for marine vessel simulations.
"""

import numpy as np
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


def ned_to_plot_angle(ned_angle):
    """
    Convert angle from NED coordinate system to matplotlib plot coordinates.
    
    NED angles: 0 = north, π/2 = east
    Plot angles: 0 = east, π/2 = north
    
    :param ned_angle: Angle in radians in NED coordinate system
    :return: Angle in radians in matplotlib plot coordinate system
    """
    return np.pi / 2 - ned_angle


def draw_ship_pose_ned(
    ax,
    pos,
    heading_ned,
    vel_dir_ned,
    radius=1.0,
    head_len=2.0,
    vel_len=2.0,
    head_color="k",
    vel_color="r",
    lw=2.5,
    zorder=10,
    draw_vel_line=True,
):
    """
    Draw ship pose visualization using NED coordinates and angle conventions.
    
    Visual elements:
    - Circle: Ship hull
    - Black line: Heading direction (bow orientation)
    - Red line: Velocity direction (actual movement direction)
    
    Coordinate mapping:
    NED: (North, East) -> plot: (x=East, y=North)
    
    Angle mapping:
    NED angles: 0 = north, π/2 = east
    Plot angles: 0 = east, π/2 = north
    
    :param ax: matplotlib.axes.Axes Axis to draw on
    :param pos: Tuple[float, float] Position (x_north, y_east) in NED coordinates
    :param heading_ned: float Heading angle ψ in radians (NED, 0 = north)
    :param vel_dir_ned: float Velocity direction angle χ in radians (NED, 0 = north)
    :param radius: float Radius of ship hull circle
    :param head_len: float Length of heading direction line
    :param vel_len: float Length of velocity direction line
    :param head_color: str Color of heading line (default: black)
    :param vel_color: str Color of velocity line (default: red)
    :param lw: float Line width for all visual elements
    :param zorder: int Drawing order (higher = drawn on top)
    :param draw_vel_line: bool Whether to draw the velocity direction line (default: True)
    :return: Tuple[Circle, Line2D, Line2D or None] (circle, heading_line, velocity_line) matplotlib objects, velocity_line is None if not drawn
    """
    x_n, y_e = pos

    # Coordinate mapping
    x_plot = y_e
    y_plot = x_n

    # Angle mapping
    theta_h = ned_to_plot_angle(heading_ned)
    theta_v = ned_to_plot_angle(vel_dir_ned)

    # Circle (ship hull)
    circle = Circle(
        (x_plot, y_plot),
        radius=radius,
        fill=False,
        edgecolor=head_color,
        linewidth=lw,
        zorder=zorder
    )
    ax.add_patch(circle)

    # Heading direction (black line)
    hx = x_plot + head_len * np.cos(theta_h)
    hy = y_plot + head_len * np.sin(theta_h)
    head_line = Line2D([x_plot, hx], [y_plot, hy],
                       color=head_color, linewidth=lw, zorder=zorder+1)
    ax.add_line(head_line)

    # Velocity direction (red line) - optionally drawn
    if draw_vel_line:
        vx = x_plot + vel_len * np.cos(theta_v)
        vy = y_plot + vel_len * np.sin(theta_v)
        vel_line = Line2D([x_plot, vx], [y_plot, vy],
                          color=vel_color, linewidth=lw, zorder=zorder+1)
        ax.add_line(vel_line)
    else:
        vel_line = None

    return circle, head_line, vel_line


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

