import numpy as np
from scipy.sparse import csr_matrix
import sys
sys.path.append("..")
from utils.compute_logical import css_logical_operator

class HGP:
    def __init__(self, h1, h2=None):
        """
        初始化超图积码。h1 和 h2 是用于超图积运算的输入矩阵。
        如果 h2 为 None，则默认 h2 = h1。
        """
        self.h1 = self.convert_to_array(h1)
        self.h2 = self.convert_to_array(h2) if h2 is not None else np.copy(self.h1)
        self.hx, self.hz = self.generate_stabilisers()
        self.lx, self.lz = css_logical_operator(self.hx, self.hz)
        self.N = self.hx.shape[1]

        Hx0 = np.zeros(self.hx.shape).astype(np.uint8)
        Hz0 = np.zeros(self.hz.shape).astype(np.uint8)
        Hx = np.vstack([self.hx, Hx0])
        Hz = np.vstack([Hz0, self.hz])
        H = np.hstack([Hx, Hz])

        self.k = self.N - np.linalg.matrix_rank(H)

    def convert_to_array(self, matrix):
        """
        将 csr_matrix 转换为 numpy array，如果输入已经是 numpy array 则直接返回。
        """
        if isinstance(matrix, csr_matrix):
            return matrix.toarray()
        return matrix

    def generate_stabilisers(self):
        """
        使用超图积生成 X 和 Z 稳定子矩阵。
        """
        hx = np.hstack([np.kron(self.h1, np.eye(self.h2.shape[1])), np.kron(np.eye(self.h1.shape[0]), self.h2.T)])
        hz = np.hstack([np.kron(np.eye(self.h1.shape[1]), self.h2), np.kron(self.h1.T, np.eye(self.h2.shape[0]))])

        return hx, hz

