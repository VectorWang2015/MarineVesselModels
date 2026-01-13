"""
Plotting utilities for marine vessel simulations.
"""

import numpy as np
import matplotlib.pyplot as plt


def add_force_direction_arrows(ax, direction_angle, arrow_length=3.0, 
                               grid_spacing=8.0, arrow_color='lightgray', 
                               alpha=0.5, linewidth=1.0, coord_system='NED'):
    """
    Add tiled arrows in the background showing direction of environmental forces.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add arrows to.
    direction_angle : float
        Direction of force in radians.
        If coord_system='NED': 0 = north, pi/2 = east.
        If coord_system='math': 0 = east, pi/2 = north.
    arrow_length : float, optional
        Length of each arrow in data units.
    grid_spacing : float, optional
        Spacing between arrow grid points in data units.
    arrow_color : str or tuple, optional
        Color of arrows.
    alpha : float, optional
        Transparency of arrows (0-1).
    linewidth : float, optional
        Line width of arrows.
    coord_system : str, optional
        Coordinate system: 'NED' (0° = north) or 'math' (0° = east).
    
    Returns
    -------
    quiver : matplotlib.quiver.Quiver
        Quiver object for further customization if needed.
    """
    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Create grid points
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_vals = np.arange(x_min, x_max, grid_spacing)
    y_vals = np.arange(y_min, y_max, grid_spacing)
    
    # Create meshgrid
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Flatten for quiver
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # Convert from NED to math coordinates if needed
    # NED: 0° = north, 90° = east
    # Math: 0° = east, 90° = north
    if coord_system == 'NED':
        # Convert NED to math: math_angle = pi/2 - ned_angle
        math_angle = np.pi/2 - direction_angle
    else:
        math_angle = direction_angle
    
    # Calculate arrow components (u, v) based on direction angle
    # Matplotlib quiver: (u, v) are components in data coordinates
    u = np.cos(math_angle) * arrow_length
    v = np.sin(math_angle) * arrow_length
    
    # Create arrays of same direction for all arrows
    u_flat = np.full_like(x_flat, u)
    v_flat = np.full_like(y_flat, v)
    
    # Add quiver plot
    quiver = ax.quiver(x_flat, y_flat, u_flat, v_flat,
                       color=arrow_color, alpha=alpha,
                       width=linewidth, headwidth=5, headlength=6,
                       headaxislength=4.5, scale=1.0, scale_units='xy',
                       angles='xy')
    
    # Restore axis limits (quiver might have changed them)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Ensure aspect ratio is preserved
    ax.set_aspect('equal')
    
    return quiver


def add_single_force_arrow(ax, x, y, direction_angle, arrow_length=4.0,
                           arrow_color='red', alpha=1.0, linewidth=2.5,
                           label=None, coord_system='NED'):
    """
    Add a single force arrow at a specific location.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add arrow to.
    x, y : float
        Starting position of arrow.
    direction_angle : float
        Direction of force in radians.
        If coord_system='NED': 0 = north, pi/2 = east.
        If coord_system='math': 0 = east, pi/2 = north.
    arrow_length : float, optional
        Length of arrow in data units.
    arrow_color : str or tuple, optional
        Color of arrow.
    alpha : float, optional
        Transparency of arrow (0-1).
    linewidth : float, optional
        Line width of arrow.
    label : str, optional
        Label for legend.
    coord_system : str, optional
        Coordinate system: 'NED' (0° = north) or 'math' (0° = east).
    
    Returns
    -------
    quiver : matplotlib.quiver.Quiver
        Quiver object.
    """
    # Convert from NED to math coordinates if needed
    if coord_system == 'NED':
        # Convert NED to math: math_angle = pi/2 - ned_angle
        math_angle = np.pi/2 - direction_angle
    else:
        math_angle = direction_angle
    
    u = np.cos(math_angle) * arrow_length
    v = np.sin(math_angle) * arrow_length
    
    quiver = ax.quiver(x, y, u, v,
                       color=arrow_color, alpha=alpha,
                       width=linewidth, headwidth=6, headlength=7,
                       headaxislength=5.5, scale=1.0, scale_units='xy',
                       angles='xy', label=label)
    
    ax.set_aspect('equal')
    return quiver