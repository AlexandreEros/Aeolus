import cupy as cp

def simpson_2d(integrand, x, y):
    """
    2D Simpson's rule integration on GPU.

    Parameters:
    -----------
    integrand : cp.ndarray, shape (n_y, n_x)
        2D array to integrate
    x : cp.ndarray, shape (n_x,)
        x-coordinates (must be equally spaced)
    y : cp.ndarray, shape (n_y,)
        y-coordinates (must be equally spaced)

    Returns:
    --------
    result : complex
        Integrated value
    """
    n_y, n_x = integrand.shape

    # Check that we have odd number of points for Simpson's rule
    # If not, we'll use composite Simpson's or fall back to trapezoidal for the last interval

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Simpson's weights for 1D: [1, 4, 2, 4, 2, ..., 4, 1]
    def simpson_weights(n):
        if n < 3:
            return cp.ones(n) * dx / 2  # Fallback to trapezoidal

        w = cp.ones(n)
        w[1:-1:2] = 4  # Odd indices (1, 3, 5, ...)
        w[2:-1:2] = 2  # Even indices (2, 4, 6, ...)

        # Handle even n by using trapezoidal for last segment
        if n % 2 == 0:
            w[-2] = 1
            w[-1] = 1
            # Composite rule: Simpson for first n-1, then adjust
            return w / 3.0
        else:
            return w / 3.0

    # Create weight matrices
    wx = simpson_weights(n_x)
    wy = simpson_weights(n_y)

    # Apply weights: outer product creates 2D weight matrix
    weights = cp.outer(wy, wx)

    # Integrate
    result = cp.sum(integrand * weights) * dx * dy

    return result