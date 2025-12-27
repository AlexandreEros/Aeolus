import cupy as cp
from cupyx.scipy import sparse


class DifferentialOperatorsSpherical:
    def __init__(
        self,
        longitude: cp.ndarray,
        latitude: cp.ndarray,
        radius: cp.ndarray,
        inv_dists: sparse.spmatrix,
        *,
        degrees: bool | None = None,
    ):
        self.longitude = longitude
        self.latitude = latitude

        if degrees is None:
            degrees = self._looks_like_degrees(self.longitude, self.latitude)
        if degrees:
            self.longitude = cp.deg2rad(self.longitude)
            self.latitude = cp.deg2rad(self.latitude)

        self.radius = radius
        if self.radius.ndim == 0:
            self.radius = cp.full(self.latitude.shape, float(self.radius), dtype=cp.float64)
        if self.radius.shape[0] != self.latitude.shape[0]:
            raise ValueError("radius must be a scalar or an array of length N.")

        if not sparse.isspmatrix(inv_dists):
            inv_dists = sparse.csr_matrix(inv_dists)
        self.inv_dists = inv_dists.tocsr()  # Non-zero elements are inverse distances between adjacent nodes

        self.cos_lat = cp.cos(self.latitude)

        self.partial_derivative_operators = self.build_partial_derivative_operators()
        self.zonal_operator, self.meridional_operator = self.partial_derivative_operators
        self.laplacian_operator = self.build_laplacian_matrix() #self.build_spherical_laplacian_operator() #


    @staticmethod
    def _looks_like_degrees(longitude: cp.ndarray, latitude: cp.ndarray) -> bool:
        max_lon = cp.nanmax(cp.abs(longitude))
        max_lat = cp.nanmax(cp.abs(latitude))
        return (max_lon > (2.0 * cp.pi + 1e-3)) or (max_lat > (0.5 * cp.pi + 1e-3))

    @classmethod
    def from_geodesic_grid(cls, grid) -> "DifferentialOperatorsSpherical":
        return cls(
            longitude=grid.longitudes,
            latitude=grid.latitudes,
            radius=grid.radial_distances,
            inv_dists=grid.adjacency_matrix,
            degrees=False,
        )

    def build_partial_derivative_operators(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Constructs gradient operators for partial derivatives with respect to longitude and latitude.

        Returns:
        - M_lambda: scipy.sparse.csr_matrix of shape (N, N) for df/dlambda.
        - M_phi: scipy.sparse.csr_matrix of shape (N, N) for df/dphi.
        """
        N = self.latitude.shape[0]

        # Initialize lists to construct sparse matrices
        data_lambda, rows_lambda, cols_lambda = [], [], []
        data_phi, rows_phi, cols_phi = [], [], []

        for i in range(N):
            # Extract neighbor indices for vertex i
            row_start = self.inv_dists.indptr[i]
            row_end = self.inv_dists.indptr[i + 1]
            neighbors = self.inv_dists.indices[row_start:row_end]
            if neighbors.size < 2:
                continue

            # Extract differences
            delta_lambda = self.longitude[neighbors] - self.longitude[i]  # (M,)
            delta_phi = self.latitude[neighbors] - self.latitude[i]  # (M,)

            # Longitude wrapping
            delta_lambda = (delta_lambda + cp.pi) % (2 * cp.pi) - cp.pi  # Wrap to [-pi, pi]

            # Scale delta_lambda by cos(phi) to account for spherical geometry
            r_mean = (self.radius[neighbors] + self.radius[i]) / 2.0
            delta_lambda_scaled = delta_lambda * self.cos_lat[i] * r_mean
            delta_phi_scaled = delta_phi * r_mean

            # Extract weights
            w = self.inv_dists.data[row_start:row_end]  # (M,)

            # Form matrix A (M x 2)
            A = cp.vstack((delta_lambda_scaled, delta_phi_scaled)).T  # Shape (M, 2)

            # Form weight matrix W (M x M), but we'll apply weights directly
            # Compute A^T W A
            AtW = A.T * w  # Each column of A.T multiplied by w
            AtWA = AtW @ A  # Shape (2, 2)

            # # Check if AtWA is invertible
            # if cp.linalg.cond(AtWA) > 1 / cp.finfo(AtWA.dtype).eps:
            #     # Singular or ill-conditioned; skip or handle appropriately
            #     # Here, we choose to skip and leave derivatives as zero
            #     continue

            # Compute C = (A^T W A)^-1 A^T W
            C = cp.linalg.solve(AtWA, AtW)  # Shape (2, M)

            # Extract coefficients for lambda and phi
            C_lambda = C[0, :]  # Shape (M,)
            C_phi = C[1, :]  # Shape (M,)

            # Assign coefficients to the sparse matrices
            for idx, j in enumerate(neighbors):
                rows_lambda.append(i)
                cols_lambda.append(j.item())
                data_lambda.append(C_lambda[idx].item())

                rows_phi.append(i)
                cols_phi.append(j.item())
                data_phi.append(C_phi[idx].item())

            # Central point
            rows_lambda.append(i)
            cols_lambda.append(i)
            data_lambda.append(-cp.sum(C_lambda).item())  # C_i_lambda multiplies f_i

            rows_phi.append(i)
            cols_phi.append(i)
            data_phi.append(-cp.sum(C_phi).item())  # C_i_phi multiplies f_i

        # Create sparse matrices in COO format, then convert to CSR
        data_lambda = cp.asarray(data_lambda, dtype=cp.float64).ravel()
        rows_lambda = cp.asarray(rows_lambda, dtype=cp.int64).ravel()
        cols_lambda = cp.asarray(cols_lambda, dtype=cp.int64).ravel()
        data_phi = cp.asarray(data_phi, dtype=cp.float64).ravel()
        rows_phi = cp.asarray(rows_phi, dtype=cp.int64).ravel()
        cols_phi = cp.asarray(cols_phi, dtype=cp.int64).ravel()

        M_lambda = sparse.coo_matrix((data_lambda, (rows_lambda, cols_lambda)), shape=(N, N)).tocsr()
        M_phi = sparse.coo_matrix((data_phi, (rows_phi, cols_phi)), shape=(N, N)).tocsr()

        return M_lambda, M_phi


    def calculate_gradient(self, values: cp.ndarray) -> cp.ndarray:
        """
        Computes the gradient of a scalar field defined on a spherical surface.

        :param values: CuPy array of shape (N,) containing scalar values at vertices.
        :return grad: CuPy array of shape (N,2) representing (df/dlambda, df/dphi) for each vertex.
        """
        shape = values.shape
        values = values.flatten()

        # Perform matrix-vector multiplication to compute the gradients
        grad_lambda = self.zonal_operator.dot(values)  # df/dlambda for each vertex
        grad_phi = self.meridional_operator.dot(values)  # df/dphi for each vertex
        gradient = cp.stack([grad_lambda, grad_phi], axis=-1)
        return gradient.reshape(shape + (2,))


    def calculate_divergence(self, vector_field: cp.ndarray) -> cp.ndarray:
        """
        Computes the divergence of a vector field defined on a spherical surface.

        Parameters:
        - vector_field: CuPy array of shape (..., 2) representing the vector field (v_lambda, v_phi) at each vertex.
        Returns:
        - divergence: Flat CuPy array representing the divergence of the vector field at each vertex.
        """
        shape = vector_field.shape
        vector_field = vector_field.reshape((-1, 2))

        # Split the vector field into components
        v_lambda = vector_field[:, 0]  # Zonal component
        v_phi = vector_field[:, 1]  # Meridional component

        # Compute divergence using matrix-vector multiplication
        div_lambda = self.zonal_operator.dot(v_lambda)  # Partial derivative of v_lambda with respect to lambda
        div_phi = self.meridional_operator.dot(v_phi)  # Partial derivative of v_phi with respect to phi
        divergence = div_lambda + div_phi
        return divergence.reshape(shape[:-1])


    def calculate_curl(self, vector_field: cp.ndarray) -> cp.ndarray:
        """
        Computes the curl of a vector field defined on a spherical surface.
        Parameters:
        - vector_field: CuPy array of shape (..., 2) representing the vector field (v_lambda, v_phi) at each vertex.
        Returns:
        - curl: CuPy array of shape (...) representing the normal component of curl.
        """
        shape = vector_field.shape
        vector_field = vector_field.reshape((-1,2))

        # Split the vector field into components
        v_lambda = vector_field[:, 0]  # Zonal component
        v_phi = vector_field[:, 1]  # Meridional component

        # Compute partial derivatives
        curl_h = self.zonal_operator.dot(v_phi) - self.meridional_operator.dot(v_lambda)  # dv_phi/dlambda - dv_lambda/dphi
        return curl_h.reshape(shape[:-1])


    def calculate_vector_gradient(self, vector_field: cp.ndarray) -> cp.ndarray:
        """
        Computes the gradient of a vector field defined on a spherical surface.

        :param vector_field: CuPy array of shape (..., 2) containing the vector field components
                             (v_lambda, v_phi) at each vertex.
        :return: CuPy array of shape (N, 2, 2) representing the gradient tensor with
                 dv/dlambda and dv/dphi for each vertex.
        """
        shape = vector_field.shape
        vector_field = vector_field.reshape((-1, 2))
        N = vector_field.shape[0]
        gradient_tensor = cp.zeros((N, 2, 2))

        # Split the vector field into zonal and meridional components
        v_lambda = vector_field[:, 0]  # Zonal component of the vector field
        v_phi = vector_field[:, 1]  # Meridional component of the vector field

        # Compute partial derivatives
        gradient_tensor[:,0,0] = self.zonal_operator.dot(v_lambda)  # dv_lambda/dlambda
        gradient_tensor[:,0,1] = self.meridional_operator.dot(v_lambda)  # dv_lambda/dphi

        gradient_tensor[:,1,0] = self.zonal_operator.dot(v_phi)  # dv_phi/dlambda
        gradient_tensor[:,1,1] = self.meridional_operator.dot(v_phi)  # dv_phi/dphi

        return gradient_tensor.reshape(shape + (2,))


    def build_laplacian_matrix(self):
        """
        Given the weighed adjacency matrix, whose weights are inverse distances, build the corresponding Laplacian matrix.
        """
        # Calculate the degree matrix as the sum of each row
        row_sum = cp.array(self.inv_dists.sum(axis=1))
        D = sparse.diags(row_sum.ravel(), format='csr')

        # Compute the Laplacian
        L = D - self.inv_dists
        return L

    def build_spherical_laplacian_operator(self) -> sparse.csr_matrix:
        """
        Build a discrete approximation of the surface Laplacian for a scalar field.

        This uses the zonal and meridional operators in local tangent-plane
        coordinates and ignores radial derivatives.
        """
        M_lambda2 = self.zonal_operator @ self.zonal_operator
        M_phi2 = self.meridional_operator @ self.meridional_operator
        return M_lambda2 + M_phi2

    def calculate_laplacian(self, values: cp.ndarray) -> cp.ndarray:
        shape = values.shape
        N = self.laplacian_operator.shape[0]
        values = values.reshape((N,-1))
        laplacian = self.laplacian_operator.dot(values)
        return laplacian.reshape(shape)
