from scipy.sparse import hstack, vstack, kron, eye, csr_matrix
from ldpc import mod2
import numpy as np
import sys
sys.path.append("..")
from utils.compute_logical import non_CSS_logical_operator

class XYZ4DCode:
    def __init__(self, L):
        self.L = L
        Hx1, Hz1 = self.shor_code_stabilizers(L[0])
        Hx2, Hz2 = self.shor_code_stabilizers(L[1])

        self.hx, self.hz = self.XYZ_product_code_stabilisers(Hz1.T, Hx1, Hz2.T, Hx2)
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

    def shor_code_stabilizers(self, L):
        code_length = L**2
        Hr = self.repetition_code(L)
        Hr = Hr[0:L-1, :]
        Hx = np.zeros((L-1, code_length)).astype(int)
        Hz = np.zeros((L * (L-1), code_length)).astype(int)
        for i in range(L-1):
            Hx[i, i*L:(i+2)*L] = 1
        for i in range(L):
            Hz[i*(L-1):(i+1)*(L-1), i*L:i*L+L] = Hr

        return Hx, Hz
    
    def XYZ_product_code_stabilisers(self, H1, H2, H3, H4):
        """
        Parity-check matrix for the stabilisers of a 4D Chamon code with size L1, L2, L3,L4
        constructed as the 4D XYZ product of three repetition codes.
        """
        m1 = H1.shape[1]
        n1 = H1.shape[0]

        m2 = H2.shape[0]
        n2 = H2.shape[1]

        m3 = H3.shape[1]
        n3 = H3.shape[0]

        m4 = H4.shape[0]
        n4 = H4.shape[1]
        nA = n2
        nB = n4

        # X
        H11 = kron(eye(m1), H4.T).astype(int)
        # Y
        H12 = kron(eye(m1), H3).astype(int)
        # Z
        H13 = kron(H1.T, eye(nB)).astype(int)
        # I
        H14 = np.zeros([m1*nB, m2*m4], dtype=np.uint8)
        # I
        H15 = np.zeros([m1*nB, m2*m3], dtype=np.uint8)

        # Y
        H21 = kron(H1, eye(m4)).astype(int)
        # I
        H22 = np.zeros([nA*m4, m1*m3], dtype=np.uint8)
        # X
        H23 = kron(eye(nA), H4).astype(int)
        # Z
        H24 = kron(H2.T, eye(m4)).astype(int)
        # I
        H25 =np.zeros([nA*m4, m2*m3], dtype=np.uint8)

        # I
        H31 = np.zeros([nA*m3, m1*m4], dtype=np.uint8)
        # Z
        H32 = kron(H1, eye(m3)).astype(int)
        # X
        H33 = kron(eye(nA), H3.T).astype(int)
        # I
        H34 = np.zeros([nA*m3, m2*m4], dtype=np.uint8)
        # Y
        H35 = kron(H2.T, eye(m3)).astype(int)

        # I
        H41 = np.zeros([m2*nB, m1*m4], dtype=np.uint8)
        # I
        H42 = np.zeros([m2 * nB, m1*m3], dtype=np.uint8)
        # Z
        H43 = kron(H2, eye(nB)).astype(int)
        # Y
        H44 = kron(eye(m2), H4.T).astype(int)
        # X
        H45 = kron(eye(m2), H3).astype(int)


        Hx_part1 = hstack([H11, H12, np.zeros(H13.shape, dtype=np.uint8), H14, H15], dtype=np.uint8)
        Hz_part1 = hstack([np.zeros(H11.shape, dtype=np.uint8), H12, H13, H14, H15], dtype=np.uint8)

        Hx_part2 = hstack([H21, H22, H23, np.zeros(H24.shape, dtype=np.uint8), H25], dtype=np.uint8)
        Hz_part2 = hstack([H21, H22, np.zeros(H23.shape, dtype=np.uint8), H24, H25], dtype=np.uint8)

        Hx_part3 = hstack([H31, np.zeros(H32.shape, dtype=np.uint8), H33, H34, H35], dtype=np.uint8)
        Hz_part3 = hstack([H31, H32, np.zeros(H33.shape, dtype=np.uint8), H34, H35], dtype=np.uint8)

        Hx_part4 = hstack([H41, H42, np.zeros(H43.shape, dtype=np.uint8), H44, H45], dtype=np.uint8)
        Hz_part4 = hstack([H41, H42, H43, H44, np.zeros(H45.shape, dtype=np.uint8)], dtype=np.uint8)

        Hx_new = vstack([Hx_part1, Hx_part2, Hx_part3, Hx_part4], dtype=np.uint8)
        Hz_new = vstack([Hz_part1, Hz_part2, Hz_part3, Hz_part4], dtype=np.uint8)

        Hx_new.data = Hx_new.data % 2
        Hz_new.data = Hz_new.data % 2
        Hx_new.eliminate_zeros()
        Hz_new.eliminate_zeros()


        # H_bar = hstack([vstack([Hx_part2, Hx_part3], dtype=np.uint8), vstack([Hz_part2, Hz_part3], dtype=np.uint8)])
        H_bar = hstack([Hx_new, Hz_new])
        HGF = H_bar.toarray()

        # H = vstack([H11.T, H12.T, H13.T], dtype=np.uint8)
        H = vstack([hstack([H21, H22, H23, H24, H25], dtype=np.uint8), hstack([H31, H32, H33, H34, H35], dtype=np.uint8)])
        # H = hstack([H13.T, H43.T], dtype=np.uint8)
        ker_H = mod2.nullspace(H.toarray()).astype(int)
        print(mod2.rank(H.toarray()))
        print("H's dimension = {}".format(H.shape))
        print("Ker of H's dimension = {}".format(ker_H.shape))
        HGFrank = mod2.rank(HGF)

        print("H_bar's rank={}".format(HGFrank))
        print("码长={}".format(Hx_new.shape[1]))
        print("此部分稳定子的数量={}".format(H_bar.shape[0]))

        return csr_matrix(Hx_new).toarray(), csr_matrix(Hz_new).toarray()
