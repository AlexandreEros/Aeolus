#include <cupy/complex.cuh>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Real Spherical Harmonics Convention:
// For m > 0: Y_l^m = sqrt(2) * P_l^m(cos φ) * cos(m θ)
// For m = 0: Y_l^0 = P_l^0(cos φ)
// For m < 0: Y_l^m = sqrt(2) * P_l^|m|(cos φ) * sin(|m| θ)
//
// But we only store m >= 0, so we compute:
// Y_l^0 = P_l^0
// Y_l^m = sqrt(2) * P_l^m * cos(m θ)  [stores the m>0 part]
//
// The negative m components would be:
// Y_l^(-m) = sqrt(2) * P_l^m * sin(m θ)

extern "C" __global__
void generate_real_sph_harm_basis(const double* lat, const double* lon,
                                  complex<double>* Y_matrix,
                                  int n_points, int l_max) {
    // Y_matrix shape: (n_points, n_basis_functions)
    // Basis functions are ordered by l, then m: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)...
    // Index = l*(l+1)/2 + m
    // 
    // NOTE: We store as complex for compatibility, but imaginary parts will be zero

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_points) return;

    double theta = lon[idx];           // Azimuthal angle
    double phi = M_PI/2.0 - lat[idx];  // Colatitude
    double x_val = cos(phi);
    double somx2 = sqrt((1.0 - x_val) * (1.0 + x_val)); // sin(phi)

    // Pointers to the row for this point
    int n_basis = (l_max + 1) * (l_max + 2) / 2;
    complex<double>* row_ptr = &Y_matrix[idx * n_basis];

    // Iterate m from 0 to l_max
    for (int m = 0; m <= l_max; m++) {

        // --- 1. Compute P_m^m ---
        // P_m^m = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
        double pmm = 1.0;
        if (m > 0) {
            double fact = 1.0;
            for (int i = 1; i <= m; i++) {
                pmm *= -fact * somx2;
                fact += 2.0;
            }
        }

        // --- 2. Compute Normalization & Recursion ---
        double p_prev = pmm;    // P_{l-1}^m
        double p_curr = pmm;    // P_l^m
        double p_prev2 = 0.0;   // P_{l-2}^m

        for (int l = m; l <= l_max; l++) {
            if (l == m) {
                p_curr = pmm;
            } else if (l == m + 1) {
                p_curr = x_val * (2.0 * m + 1.0) * pmm;
            } else {
                // Recurrence: P_l^m = [(2l-1)x P_{l-1}^m - (l+m-1)P_{l-2}^m] / (l-m)
                p_curr = ((2.0 * l - 1.0) * x_val * p_prev - (l + m - 1.0) * p_prev2) / (l - m);
            }

            // Normalization: N = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! )
            double log_fact_diff = lgamma(l - m + 1.0) - lgamma(l + m + 1.0);
            double norm = sqrt( (2.0 * l + 1.0) / (4.0 * M_PI) * exp(log_fact_diff) );

            // REAL spherical harmonics:
            // - For m = 0: just P_l^0 * norm
            // - For m > 0: sqrt(2) * P_l^m * norm * cos(m*theta)
            
            double real_part;
            if (m == 0) {
                real_part = p_curr * norm;
            } else {
                real_part = sqrt(2.0) * p_curr * norm * cos(m * theta);
            }

            // Linear index
            int basis_idx = l * (l + 1) / 2 + m;
            row_ptr[basis_idx] = complex<double>(real_part, 0.0);  // Imaginary part is zero

            // Shift recurrence
            p_prev2 = p_prev;
            p_prev = p_curr;
        }
    }
}
