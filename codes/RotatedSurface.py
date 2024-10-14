import numpy as np
from qecsim.models.rotatedplanar import RotatedPlanarCode

class RotatedSurfaceCode:
    def __init__(self, L):
        self.L = L
        code = RotatedPlanarCode(L, L)
        H = code.stabilizers
        M = H.shape[0] // 2
        self.N = H.shape[1] // 2

        self.hz = H[:M, self.N:]
        self.hx = H[M:, :self.N]

        self.lx = code.logical_xs[0][:self.N]
        self.lz = code.logical_zs[0][self.N:]

        Hx0 = np.zeros(self.hx.shape).astype(np.uint8)
        Hz0 = np.zeros(self.hz.shape).astype(np.uint8)
        Hx = np.vstack([self.hx, Hx0])
        Hz = np.vstack([Hz0, self.hz])
        H = np.hstack([Hx, Hz])

        self.k = self.N - np.linalg.matrix_rank(H)

        assert np.all(np.mod(np.dot(self.hx, self.hz.T), 2) == 0), "hx * hz.T is not zero"
        assert np.all(np.mod(np.dot(self.hx, self.lz.T), 2) == 0), "hx * lz.T is not zero"
        assert np.all(np.mod(np.dot(self.hz, self.lx.T), 2) == 0), "hz * lx.T is not zero"
