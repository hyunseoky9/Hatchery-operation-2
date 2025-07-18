from typing import Tuple

def simplex_color(
    x: float,
    y: float,
    z: float,
    base_rgb: Tuple[Tuple[int, int, int],
                    Tuple[int, int, int],
                    Tuple[int, int, int]] = (
                        (255,   0,   0),   # colour for x  – red
                        (  0, 255,   0),   # colour for y  – green
                        (  0,   0, 255))   # colour for z  – blue
) -> str:
    """
    Map three positive numbers summing to 1 onto the RGB simplex.

    Parameters
    ----------
    x, y, z : float
        Weights (must all be >0 and sum to 1 within a small tolerance).
    base_rgb : 3‑tuple of 3‑tuples
        Vertex colours for (x, y, z) in 0‑255 RGB.

    Returns
    -------
    str
        Hex string like '#a1b2c3'.

    Examples
    --------
    >>> simplex_color(1, 0, 0)          # pure x (red)
    '#ff0000'
    >>> simplex_color(0.5, 0.5, 0)      # mid‑way between red & green
    '#7f7f00'
    >>> simplex_color(1/3, 1/3, 1/3)    # white-ish grey
    '#aaaaaa'
    """
    # --- validate inputs ----------------------------------------------------
    eps = 1e-9
    if min(x, y, z) <= 0:
        raise ValueError("x, y and z must all be > 0.")
    if abs((x + y + z) - 1.0) > eps:
        raise ValueError("x + y + z must equal 1 (within numerical tolerance).")

    # --- blend in RGB space -------------------------------------------------
    r = x * base_rgb[0][0] + y * base_rgb[1][0] + z * base_rgb[2][0]
    g = x * base_rgb[0][1] + y * base_rgb[1][1] + z * base_rgb[2][1]
    b = x * base_rgb[0][2] + y * base_rgb[1][2] + z * base_rgb[2][2]

    # --- convert to hex -----------------------------------------------------
    return "#{:02x}{:02x}{:02x}".format(int(round(r)), int(round(g)), int(round(b)))