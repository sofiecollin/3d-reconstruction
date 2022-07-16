import numpy as np

def test_init(E, K, corr1, corr2):
    # Test 1 - E determinant close to 0
    assert np.abs(np.linalg.det(E)) < 1e-10

    # Test 2 - sigma1 and sigma2 for E should be same, sigma 3 close to zero
    (U, S, Vt) = np.linalg.svd(E)
    assert S[1] - S[0] < 1e-10
    assert S[2] < 1e-10

    # Test 3 - E internal constraint
    assert(np.abs(np.mean(E@E.T@E - 0.5*np.trace(E.T@E)*E)) < 1e-2)

    # Test 4 - v2^T * E * v1 should give closest to zero for corresponding points
    for i in range(corr1.shape[1]):
        min_i = -1
        min_res = np.inf
        for j in range(corr1.shape[1]):
            p1 = np.append(corr1[:, i], 1)
            p2 = np.append(corr2[:, j], 1)
            v1 = np.linalg.inv(K) @ p1[:, None]
            v2 = np.linalg.inv(K) @ p2[:, None]

            res = np.abs(v1.T @ E @ v2)
            if res < min_res:
                min_res = res
                min_i = j

        assert(min_i == i)

    # Test 5
    # ...