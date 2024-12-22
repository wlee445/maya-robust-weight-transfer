import igl
import numpy as np
import scipy as sp


def find_closest_point_on_surface(P, V, F):
    """
    Given a number of points find their closest points on the surface of the V,F mesh

    Args:
        P: #P by 3, where every row is a point coordinate
        V: #V by 3 mesh vertices
        F: #F by 3 mesh triangles indices
    Returns:
        sqrD #P smallest squared distances
        I #P primitive indices corresponding to smallest distances
        C #P by 3 closest points
        B #P by 3 of the barycentric coordinates of the closest point
    """

    sqrD, I, C = igl.point_mesh_squared_distance(P, V, F)

    F_closest = F[I, :]
    V1 = V[F_closest[:, 0], :]
    V2 = V[F_closest[:, 1], :]
    V3 = V[F_closest[:, 2], :]

    B = igl.barycentric_coordinates_tri(C, V1, V2, V3)

    return sqrD, I, C, B


def interpolate_attribute_from_bary(A, B, I, F):
    """
    Interpolate per-vertex attributes A via barycentric coordinates B of the F[I,:] vertices

    Args:
        A: #V by N per-vertex attributes
        B  #B by 3 array of the barycentric coordinates of some points
        I  #B primitive indices containing the closest point
        F: #F by 3 mesh triangle indices
    Returns:
        A_out #B interpolated attributes
    """
    F_closest = F[I, :]
    a1 = A[F_closest[:, 0], :]
    a2 = A[F_closest[:, 1], :]
    a3 = A[F_closest[:, 2], :]

    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]

    b1 = b1.reshape(-1, 1)
    b2 = b2.reshape(-1, 1)
    b3 = b3.reshape(-1, 1)

    A_out = a1 * b1 + a2 * b2 + a3 * b3

    return A_out


def normalize_vec(v):
    return v / np.linalg.norm(v)


def find_matches_closest_surface(source_verts, source_triangles, source_normals, target_verts, target_normals,
                                 source_weights, dDISTANCE_THRESHOLD_SQRD, dANGLE_THRESHOLD_DEGREES,
                                 flip_vertex_normal):
    """
    For each vertex on the target mesh find a match on the source mesh.

    Args:
        V1: #V1 by 3 source mesh vertices
        F1: #F1 by 3 source mesh triangles indices
        N1: #V1 by 3 source mesh normals

        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        N2: #V2 by 3 target mesh normals

        W1: #V1 by num_bones source mesh skin weights

        dDISTANCE_THRESHOLD_SQRD: scalar distance threshold
        dANGLE_THRESHOLD_DEGREES: scalar normal threshold

    Returns:
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
        W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
    """
    sqrD, I, C, B = find_closest_point_on_surface(target_verts, source_verts, source_triangles)

    # for each closest point on the source, interpolate its per-vertex attributes(skin weights and normals)
    # using the barycentric coordinates
    W2 = interpolate_attribute_from_bary(source_weights, B, I, source_triangles)
    N1_match_interpolated = interpolate_attribute_from_bary(source_normals, B, I, source_triangles)

    norm_N1 = np.linalg.norm(N1_match_interpolated, axis=1, keepdims=True)
    norm_N2 = np.linalg.norm(target_normals, axis=1, keepdims=True)
    normalized_N1 = N1_match_interpolated / norm_N1
    normalized_N2 = target_normals / norm_N2

    dot_product = np.einsum('ij,ij->i', normalized_N1, normalized_N2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure the dot product is in the valid range for arccos
    rad_angles = np.arccos(dot_product)
    deg_angles = np.degrees(rad_angles)
    is_distance_threshold = sqrD <= dDISTANCE_THRESHOLD_SQRD
    angle_thresholds = np.full(deg_angles.shape, dANGLE_THRESHOLD_DEGREES)

    is_deg_threshold = deg_angles <= angle_thresholds
    if flip_vertex_normal:
        deg_angles_mirror = 180 - deg_angles
        is_deg_threshold = np.logical_or(is_deg_threshold, deg_angles_mirror <= angle_thresholds)

    Matched = np.logical_and(is_distance_threshold, is_deg_threshold)
    return Matched, W2


def compute_cotangent(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    compute cotangent from three points.
    """
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = np.cross(edge1, edge2)
    area = np.linalg.norm(normal)
    return np.dot(edge1, edge2) / area


def add_laplacian_entry_in_place(L, tri_positions, tri_indices):
    """
    add laplacian entry.
    CAUTION: L is modified in-place.
    """

    i1 = tri_indices[0]
    i2 = tri_indices[1]
    i3 = tri_indices[2]

    v1 = tri_positions[0]
    v2 = tri_positions[1]
    v3 = tri_positions[2]

    # calculate cotangent
    cotan1 = compute_cotangent(v2, v1, v3)
    cotan2 = compute_cotangent(v1, v2, v3)
    cotan3 = compute_cotangent(v1, v3, v2)

    # update laplacian matrix
    L[i1, i2] += cotan1  # type: ignore
    L[i2, i1] += cotan1  # type: ignore
    L[i1, i1] -= cotan1  # type: ignore
    L[i2, i2] -= cotan1  # type: ignore

    L[i2, i3] += cotan2  # type: ignore
    L[i3, i2] += cotan2  # type: ignore
    L[i2, i2] -= cotan2  # type: ignore
    L[i3, i3] -= cotan2  # type: ignore

    L[i1, i3] += cotan3  # type: ignore
    L[i3, i1] += cotan3  # type: ignore
    L[i1, i1] -= cotan3  # type: ignore
    L[i3, i3] -= cotan3  # type: ignore


def add_area_in_place(areas, tri_positions, tri_indices):
    """add area.

    CAUTION: areas is modified in-place.
    """

    v1 = tri_positions[0]
    v2 = tri_positions[1]
    v3 = tri_positions[2]
    area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    for idx in tri_indices:
        areas[idx] += area


def compute_laplacian_and_mass_matrix(vertices: np.ndarray, triangles: np.ndarray):
    """
    compute laplacian matrix from mesh.
    treat area as mass matrix.
    """

    # initialize sparse laplacian matrix
    n_vertices = len(vertices)
    L = sp.sparse.lil_matrix((n_vertices, n_vertices))
    areas = np.zeros(n_vertices)

    # for each edge and face, calculate the laplacian entry and area
    for tri in triangles:
        tri_positions = np.array([vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]])
        tri_indices = tri
        add_laplacian_entry_in_place(L, tri_positions, tri_indices)
        add_area_in_place(areas, tri_positions, tri_indices)
    L_csr = L.tocsr()
    M_csr = sp.sparse.diags(areas)
    return L_csr, M_csr


def inpaint(V2, F2, W2, Matched, point_cloud):
    """
    Inpaint weights for all the vertices on the target mesh for which  we didnt
    find a good match on the source (i.e. Matched[i] == False).

    Args:
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh

    Returns:
        W_inpainted: #V2 by num_bones, final skinning weights where we inpainted weights for all vertices i where Matched[i] == False
    """
    L, M = compute_laplacian_and_mass_matrix(V2, F2)
    L = -L  # igl and robust_laplacian have different laplacian conventions

    Minv = sp.sparse.diags(1 / M.diagonal())  # divide by zero?

    Q2 = -L + L * Minv * L
    Q2 = Q2.astype(np.float64)

    Aeq = sp.sparse.csc_matrix((0, 0), dtype=np.float64)
    Beq = np.array([], dtype=np.float64)
    B = np.zeros(shape=(L.shape[0], W2.shape[1]), dtype=np.float64)

    b = np.array(range(0, int(V2.shape[0])), dtype=np.int64)
    b = b[Matched]
    bc = W2[Matched, :].astype(np.float64)
    result, W_inpainted = igl.min_quad_with_fixed(Q2, B, b, bc, Aeq, Beq, True)
    W_inpainted = W_inpainted.astype(np.float32)
    return result, W_inpainted  # TODO: Add results


def limit_mask(weights, adjacency_matrix, bone_limit_num=4, dilation_repeat=5):
    if weights.shape[1] <= bone_limit_num: return np.zeros_like(weights)

    count = np.count_nonzero(weights, axis=1)
    to_limit = count > bone_limit_num
    k = weights.shape[1] - bone_limit_num
    weights_inds = np.argpartition(weights, kth=k, axis=1)[:, :k]
    row_indices = np.arange(weights.shape[0])[:, None]
    erode_mask = np.zeros_like(weights, dtype=bool)
    erode_mask[row_indices, weights_inds] = True
    erode_mask = np.logical_and(erode_mask, to_limit[:, np.newaxis])
    erode_mask = sp.sparse.csr_array(erode_mask).astype(np.float32)
    adj_mat = adjacency_matrix
    degrees = adj_mat.sum(axis=1)
    smooth_mat = (1 / degrees[:, np.newaxis]) * adj_mat
    for _ in range(dilation_repeat):
        avg_weights = smooth_mat @ erode_mask
        erode_mask = erode_mask.maximum(avg_weights)

    return erode_mask.toarray()


def get_adjacency_data(vertices, edge):
    num_verts = len(vertices)
    rows = np.hstack([edge[:, 0], edge[:, 1]])
    cols = np.hstack([edge[:, 1], edge[:, 0]])
    data = np.ones(len(rows), dtype=int)  # Corresponding data entries for the CSR matrix

    # Create a symmetric adjacency matrix (since each edge is undirected)
    adjacency_matrix = sp.sparse.csr_array((data, (rows, cols)), shape=(num_verts, num_verts))
    adjacency_matrix.setdiag(1)

    adj_list = [[] for _ in range(num_verts)]
    for edge in edge:
        adj_list[edge[0]].append(edge[1])
        adj_list[edge[1]].append(edge[0])

    return adjacency_matrix, adj_list


def smooth_weigths(verts, weights, matched, adjacency_matrix, adjacency_list, num_smooth_iter_steps, smooth_alpha,
                   distance_threshold):
    not_matched = ~matched
    VIDs_to_smooth = np.zeros(verts.shape[0], dtype=bool)

    def get_points_within_distance(V, VID, distance=distance_threshold):
        """
        Get all neighbours of vertex VID within dDISTANCE_THRESHOLD
        """
        queue = []
        queue.append(VID)
        while len(queue) != 0:
            vv = queue.pop()
            if vv < len(adjacency_list):
                neigh = adjacency_list[vv]
                for nn in neigh:
                    if ~VIDs_to_smooth[nn] and np.linalg.norm(V[VID, :] - V[nn]) < distance:
                        VIDs_to_smooth[nn] = True
                        if nn not in queue:
                            queue.append(nn)

    for i in range(verts.shape[0]):
        if not_matched[i]:
            get_points_within_distance(verts, i, distance_threshold)

    adj_mat = adjacency_matrix.astype(np.float32)
    degrees = adj_mat.sum(axis=1)

    smooth_mat = sp.sparse.diags(1 / degrees) @ adj_mat
    weights_smoothed = sp.sparse.csr_array(weights)
    for _ in range(num_smooth_iter_steps):
        weights_smoothed = (1 - smooth_alpha) * weights_smoothed + smooth_alpha * (smooth_mat @ weights_smoothed)
        weights_smoothed[~VIDs_to_smooth] = weights[~VIDs_to_smooth]
    return weights_smoothed.todense()
