#include <cupy/complex.cuh>

extern "C" __global__
void compute_sph_harm(const double* theta, const double* phi,
                      const double* P_lm_pre_phased_normalized, complex<double>* Y,
                      int n_points, int l, int m) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_points) return;

    double theta_val = theta[idx];
    double P_val = P_lm_pre_phased_normalized[idx]; // This P_val already includes phase and normalization

    // Y_l^m = P_l^m(cos(phi))_pre_phased_normalized * e^(i m theta)
    double cos_mtheta = cos(m * theta_val);
    double sin_mtheta = sin(m * theta_val);

    Y[idx] = complex<double>(P_val * cos_mtheta,
                             P_val * sin_mtheta);
}