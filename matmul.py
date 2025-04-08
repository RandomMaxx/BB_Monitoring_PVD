@njit(cache=True)
def numba_batch_matmul_same(m1, m2):
    _, _, N = m1.shape
    result = np.empty((2, 2, N), dtype=m1.dtype)
    for i in range(N):
        a = m1[:, :, i]
        b = m2[:, :, i]
        result[0, 0, i] = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0]
        result[0, 1, i] = a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1]
        result[1, 0, i] = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0]
        result[1, 1, i] = a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1]
    return result

@njit(cache=True)
def numba_batch_matmul_single(m1, m2):
    _, _, N = m1.shape
    result = np.empty((2, 2, N), dtype=m1.dtype)
    for i in range(N):
        a = m1[:, :, i]
        result[0, 0, i] = a[0, 0] * m2[0, 0] + a[0, 1] * m2[1, 0]
        result[0, 1, i] = a[0, 0] * m2[0, 1] + a[0, 1] * m2[1, 1]
        result[1, 0, i] = a[1, 0] * m2[0, 0] + a[1, 1] * m2[1, 0]
        result[1, 1, i] = a[1, 0] * m2[0, 1] + a[1, 1] * m2[1, 1]
    return result

def matmul_auto(m1, m2):
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    if m1.shape[:2] != (2, 2):
        raise ValueError(f"m1 must be shape (2, 2, N); got {m1.shape}")
    
    if m1.ndim != 3:
        raise ValueError(f"m1 must be 3D (2, 2, N); got {m1.shape}")
    
    if m2.shape == (2, 2):  # single matrix
        return numba_batch_matmul_single(m1, m2)
    
    elif m2.shape == m1.shape:
        return numba_batch_matmul_same(m1, m2)

    else:
        raise ValueError(f"Unsupported shapes: m1 {m1.shape}, m2 {m2.shape}")
