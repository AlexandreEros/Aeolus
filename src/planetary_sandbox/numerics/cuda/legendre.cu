extern "C" __global__
void compute_legendre(const double* x, double* P, int n_points, int l, int m) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_points) return;

    double x_val = x[idx];
    double pmm = 1.0;

    // Compute P_m^m using the formula: P_m^m = (-1)^m (2m-1)!! (1-x^2)^(m/2)
    if (m > 0) {
        double somx2 = sqrt((1.0 - x_val) * (1.0 + x_val)); // sqrt(1-x^2), more stable
        double fact = 1.0;
        for (int i = 1; i <= m; i++) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }

    if (l == m) {
        P[idx] = pmm;
        return;
    }

    // Compute P_{m+1}^m
    double pmmp1 = x_val * (2.0 * m + 1.0) * pmm;

    if (l == m + 1) {
        P[idx] = pmmp1;
        return;
    }

    // Use recurrence relation for l > m+1
    // P_l^m = [(2l-1) x P_{l-1}^m - (l+m-1) P_{l-2}^m] / (l-m)
    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ll++) {
        pll = ((2.0 * ll - 1.0) * x_val * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }

    P[idx] = pll;
}