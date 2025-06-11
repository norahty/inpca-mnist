import numpy as np

def inpca_embed_intensive(P, n_components=3):
    """
    True InPCA embedding as in Quinn et al.
      P : (N, 10) soft-max matrix
      returns coords (N, n_components)
    """
    # 1. Hellinger map
    Z = np.sqrt(P)                       # shape (N,10)

    # 2. Pairwise overlaps  H_ij = <√p_i, √p_j>
    H = Z @ Z.T                          # (N,N)

    # 3. Intensive log-similarity matrix
    L = 4.0 * np.log(np.clip(H, 1e-12, 1.0))

    # 4. Double-centre   W = J L J
    N = L.shape[0]
    J = np.eye(N) - np.ones((N, N)) / N
    W = J @ L @ J

    # 5. Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(W)
    idx = eigvals.argsort()[::-1]        # largest first
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # keep only positive (or the largest-magnitude) eigenpair, NEED TO CHANGE TO LARGEST MAGNITUDE
    keep = eigvals > 0
    eigvals, eigvecs = eigvals[keep][:n_components], eigvecs[:, keep][:, :n_components]

    # 6. Coordinates   T = U √Σ
    coords = eigvecs * np.sqrt(eigvals)
    return coords, eigvals