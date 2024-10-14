from scipy.sparse import hstack, kron, eye, block_diag, csr_matrix
import numpy as np

class ToricCode:
    def __init__(self, L):
        self.L = L
        self.N = 2 * L ** 2
        self.k = 2
        self.hx = self.generate_x_stabilisers()
        self.hz = self.generate_z_stabilisers()
        self.lx = self.generate_x_logicals()
        self.lz = self.generate_z_logicals()

    def repetition_code(self, n):
        """
        Parity check matrix of a repetition code with length n.
        """
        row_ind, col_ind = zip(*((i, j) for i in range(n) for j in (i, (i+1)%n)))
        data = np.ones(2*n, dtype=np.uint8)
        return csr_matrix((data, (row_ind, col_ind))).toarray()

    def generate_x_stabilisers(self):
        """
        生成Toric码的X稳定子矩阵
        """
        Hr = self.repetition_code(self.L)
        H = hstack(
            [kron(Hr, eye(Hr.shape[1])), kron(eye(Hr.shape[0]), Hr.T)],
            dtype=np.uint8
        )
        H.data = H.data % 2
        H.eliminate_zeros()
        return csr_matrix(H).toarray()

    def generate_z_stabilisers(self):
        """
        生成Toric码的Z稳定子矩阵
        """
        Hr = self.repetition_code(self.L)
        H = hstack(
            [kron(eye(Hr.shape[1]), Hr), kron(Hr.T, eye(Hr.shape[0]))],
            dtype=np.uint8
        )
        H.data = H.data % 2
        H.eliminate_zeros()
        return csr_matrix(H).toarray()

    def generate_x_logicals(self):
        """
        生成Toric码的X逻辑操作符矩阵
        """
        H1 = csr_matrix(([1], ([0], [0])), shape=(1, self.L), dtype=np.uint8)
        H0 = csr_matrix(np.ones((1, self.L), dtype=np.uint8))
        x_logicals = block_diag([kron(H1, H0), kron(H0, H1)])
        x_logicals.data = x_logicals.data % 2
        x_logicals.eliminate_zeros()
        return csr_matrix(x_logicals).toarray()

    def generate_z_logicals(self):
        """
        生成Toric码的Z逻辑操作符矩阵
        """
        H1 = csr_matrix(([1], ([0], [0])), shape=(1, self.L), dtype=np.uint8)
        H0 = csr_matrix(np.ones((1, self.L), dtype=np.uint8))
        z_logicals = block_diag([kron(H0, H1), kron(H1, H0)])
        z_logicals.data = z_logicals.data % 2
        z_logicals.eliminate_zeros()
        return csr_matrix(z_logicals).toarray()
