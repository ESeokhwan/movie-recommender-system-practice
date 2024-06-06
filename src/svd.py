import numpy as np


def full_svd(rating_matrix: np.ndarray) -> np.ndarray:
    U, s_diag, Vt = np.linalg.svd(rating_matrix)
    print("single value decomposition")
    print(f"U: {U.shape}, s: {s_diag.shape}, Vt: {Vt.shape}")

    s = np.zeros((U.shape[1], Vt.shape[0]))
    s[: s_diag.shape[0], : s_diag.shape[0]] = np.diag(s_diag)

    return U @ s @ Vt


def truncated_svd(rating_matrix: np.ndarray, k: int) -> np.ndarray:
    U, s_diag, Vt = np.linalg.svd(rating_matrix)
    print("truncated single value decomposition")

    U = U[:, :k]
    s_diag = s_diag[:k]
    Vt = Vt[:k, :]
    print(f"U: {U.shape}, s: {s_diag.shape}, Vt: {Vt.shape}")

    s = np.zeros((U.shape[1], Vt.shape[0]))
    s[: s_diag.shape[0], : s_diag.shape[0]] = np.diag(s_diag)

    return U @ s @ Vt
