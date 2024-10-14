from scipy.sparse import hstack, vstack, kron, eye, csr_matrix
from ldpc import mod2
import numpy as np
import sys
sys.path.append("..")
from utils.compute_logical import non_CSS_logical_operator

class XYZ3DCode:
    def __init__(self, L):
        self.L = L
        RPcode = self.repetition_code(L)
        self.hx, self.hz = self.XYZ_product_code_stabilisers(RPcode, RPcode, RPcode)
        H = np.hstack([self.hx, self.hz])
        self.lx, self.lz = non_CSS_logical_operator(H)
        self.N = self.hx.shape[1]
        self.k = self.N - np.linalg.matrix_rank(H)


    def repetition_code(self, n):
        """
        Parity check matrix of a repetition code with length n.
        """
        row_ind, col_ind = zip(*((i, j) for i in range(n) for j in (i, (i+1)%n)))
        data = np.ones(2*n, dtype=np.uint8)
        return csr_matrix((data, (row_ind, col_ind))).toarray()
    
    def XYZ_product_code_stabilisers(self, H1, H2, H3):
        """
        Parity-check matrix for the stabilisers of a 3D Chamon code with size L1, L2, L3,
        constructed as the XYZ product of three repetition codes.
        """
        # H1 = repetition_code(L1)
        # H2 = repetition_code(L2)
        # H3 = repetition_code(L3)

        m1 = H1.shape[0]
        n1 = H1.shape[1]
        m2 = H2.shape[0]
        n2 = H2.shape[1]
        m3 = H3.shape[0]
        n3 = H3.shape[1]

        # X
        H11 = kron(kron(H1.T, eye(n2)), eye(n3))
        # Y
        H21 = kron(kron(eye(m1), H2), eye(n3))
        # Z
        H31 = kron(kron(eye(m1), eye(n2)), H3)

        H41 = np.zeros([n1*m2*m3, m1*n2*n3], dtype=np.uint8)

        # Y
        H12 = kron(kron(eye(n1), H2.T), eye(n3))
        # X
        H22 = kron(H1, kron(eye(m2), eye(n3)))

        H32 = np.zeros([m1*n2*n3, n1*m2*n3], dtype=np.uint8)
        # Z
        H42 = kron(kron(eye(n1), eye(m2)), H3)

        # Z
        H13 = kron(kron(eye(n1), eye(n2)), H3.T)
        H23 = np.zeros([m1*m2*n3, n1*n2*m3], dtype=np.uint8)
        # X
        H33 = kron(H1, kron(eye(n2), eye(m3)))
        # Y
        H43 = kron(kron(eye(n1), H2), eye(m3))

        H14 = np.zeros([n1*n2*n3, m1*m2*m3], dtype=np.uint8)
        # Z
        H24 = kron(kron(eye(m1), eye(m2)), H3.T)
        # Y
        H34 = kron(kron(eye(m1), H2.T), eye(m3))
        # X
        H44 = kron(H1.T, kron(eye(m2), eye(m3)))

        Hx_part1 = vstack([H11, H21, np.zeros(H31.shape, dtype=np.uint8), H41], dtype=np.uint8).T
        Hz_part1 = vstack([np.zeros(H11.shape, dtype=np.uint8), H21, H31, H41], dtype=np.uint8).T

        Hx_part2 = vstack([H12, H22, H32, np.zeros(H42.shape, dtype=np.uint8)], dtype=np.uint8).T
        Hz_part2 = vstack([H12, np.zeros(H22.shape, dtype=np.uint8), H32, H42], dtype=np.uint8).T

        Hx_part3 = vstack([np.zeros(H13.shape, dtype=np.uint8), H23, H33, H43], dtype=np.uint8).T
        Hz_part3 = vstack([H13, H23, np.zeros(H33.shape, dtype=np.uint8), H43], dtype=np.uint8).T

        Hx_part4 = vstack([H14, np.zeros(H24.shape, dtype=np.uint8), H34, H44], dtype=np.uint8).T
        Hz_part4 = vstack([H14, H24, H34, np.zeros(H44.shape, dtype=np.uint8)], dtype=np.uint8).T

        Hx = vstack([Hx_part1, Hx_part2, Hx_part3, Hx_part4], dtype=np.uint8)
        Hz = vstack([Hz_part1, Hz_part2, Hz_part3, Hz_part4], dtype=np.uint8)

        Hx.data = Hx.data % 2
        Hz.data = Hz.data % 2
        Hx.eliminate_zeros()
        Hz.eliminate_zeros()

        # Hx = Hx.T
        # Hz = Hz.T

        # H_bar = hstack([Hx, Hz]).toarray()
        H_bar = hstack([vstack([Hx_part1, Hx_part2, Hx_part4], dtype=np.uint8), vstack([Hz_part1, Hz_part2, Hz_part4], dtype=np.uint8)])

        print("H's rank={}".format(mod2.rank(H_bar.toarray())))

        return csr_matrix(Hx).toarray(), csr_matrix(Hz).toarray()