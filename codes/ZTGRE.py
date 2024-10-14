import numpy as np

class ZTGRE:
    def __init__(self, iterations):
        """
        初始化 ZTGRE 类，执行递归构建 Hz, Lx, Lz 矩阵。
        """
        self.hz, self.lx, self.lz = self.ZTGRE_construction(iterations)
        self.hx = np.zeros(self.hz.shape)
        self.N = self.hz.shape[1]
        self.k = self.lx.shape[0]

    def ZTGRE_construction(self, iter):
        def recursive_construction(iter):
            if iter == 0:
                # Recursive base case
                Hl = np.zeros((1, 2))
                Hl[0, [0, 1]] = 1
                pivot = np.zeros(1, dtype=int)
                pivot[0] = 1

            else:
                Hl_pre, pivot_pre = recursive_construction(iter - 1)
                m_pre, n_pre = Hl_pre.shape
                Hl = np.zeros((m_pre * 2, n_pre * 2))

                # Insert previous Hl matrix
                Hl[0:m_pre, 0:n_pre] = Hl_pre

                # Compute and insert the regular elements of the new check matrix's dual rows
                for row in range(m_pre):
                    nonZeroColumnsHl = np.where(Hl_pre[row, :] != 0)[0]
                    newColumnIndicesHl = nonZeroColumnsHl + n_pre
                    Hl[row + m_pre, newColumnIndicesHl] = Hl_pre[row, nonZeroColumnsHl]

                # Update pivot coordinates
                pivot = np.zeros(pivot_pre.size * 2, dtype=int)
                pivot[0:pivot_pre.size] = pivot_pre
                pivot[pivot_pre.size:] = pivot_pre + n_pre

                # Insert the connecting elements of the dual rows
                for row in range(m_pre * 2):
                    pos_l = (pivot[row] + n_pre) % (n_pre * 2)
                    pos_l = n_pre * 2 if pos_l == 0 else pos_l

                    Hl[row, pos_l - 1] = 1

            return Hl, pivot
        
        Hz, _ = recursive_construction(iter)

        def compute_Lx(iter):
            Hz, _ = recursive_construction(iter)
            if iter % 2 == 0:
                Lx = Hz.copy()
            else:
                Lx = np.zeros_like(Hz)
                num_rows, num_cols = Hz.shape
                for row in range(num_rows):
                    indices = np.where(Hz[row, :] != 0)[0]
                    new_indices = []
                    for i in indices:
                        if i % 2 == 0:  # even index (zero-based)
                            i_new = i + 1
                        else:           # odd index
                            i_new = i - 1
                        if 0 <= i_new < num_cols:
                            new_indices.append(i_new)
                    Lx[row, new_indices] = 1
            return Lx

        def compute_Lz(iter):
            rows = 2 ** iter
            columns = 2 * rows
            Lz = np.zeros((rows, columns))
            offset = (iter + 1) % 2

            if iter == 0:
                Lz[0, 0] = 1
                return Lz

            for r in range(rows):
                c = (2 * r + offset) % columns
                Lz[r, c] = 1

            return Lz
        
        Lx = compute_Lx(iter)
        Lz = compute_Lz(iter)

        # print("Hz stabilizers:")
        # for r in range(Hz.shape[0]):
        #     nonZeroElementsHz = np.where(Hz[r, :] != 0)[0] + 1
        #     print(f"Row {r + 1}: {nonZeroElementsHz}")

        # Lx = np.zeros((2**iter, 2*2**iter))
        # Lz = np.zeros((2**iter, 2*2**iter))

        # if iter == 0:
        #     Lx[0, [0, 1]] = 1
        #     Lz[0, 0] = 1

        # elif iter == 1:
        #     Lx[0, [0, 1, 3]] = 1
        #     Lx[1, [1, 2, 3]] = 1

        #     Lz[0, 0] = 1
        #     Lz[1, 2] = 1

        # elif iter == 2:
        #     Lx[0, [0, 1, 2, 4]] = 1
        #     Lx[1, [0, 2, 3, 6]] = 1
        #     Lx[2, [0, 4, 5, 6]] = 1
        #     Lx[3, [2, 4, 6, 7]] = 1

        #     Lz[0, 1] = 1
        #     Lz[1, 3] = 1
        #     Lz[2, 5] = 1
        #     Lz[3, 7] = 1

        # elif iter == 3:
        #     Lx[0, [0, 1, 3, 5, 9]] = 1
        #     Lx[1, [1, 2, 3, 7, 11]] = 1
        #     Lx[2, [1, 4, 5, 7, 13]] = 1
        #     Lx[3, [3, 5, 6, 7, 15]] = 1
        #     Lx[4, [1, 8, 9, 11, 13]] = 1
        #     Lx[5, [3, 9, 10, 11, 15]] = 1
        #     Lx[6, [5, 9, 12, 13, 15]] = 1
        #     Lx[7, [7, 11, 13, 14, 15]] = 1

        #     Lz[0, 0] = 1
        #     Lz[1, 2] = 1
        #     Lz[2, 4] = 1
        #     Lz[3, 6] = 1
        #     Lz[4, 8] = 1
        #     Lz[5, 10] = 1
        #     Lz[6, 12] = 1
        #     Lz[7, 14] = 1

        # elif iter == 4:
        #     Lx[0, [0, 1, 2, 4, 8, 16]] = 1
        #     Lx[1, [0, 2, 3, 6, 10, 18]] = 1
        #     Lx[2, [0, 4, 5, 6, 12, 20]] = 1
        #     Lx[3, [2, 4, 6, 7, 14, 22]] = 1
        #     Lx[4, [0, 8, 9, 10, 12, 24]] = 1
        #     Lx[5, [2, 8, 10, 11, 14, 26]] = 1
        #     Lx[6, [4, 8, 12, 13, 14, 28]] = 1
        #     Lx[7, [6, 10, 12, 14, 15, 30]] = 1
        #     Lx[8, [0, 16, 17, 18, 20, 24]] = 1
        #     Lx[9, [2, 16, 18, 19, 22, 26]] = 1
        #     Lx[10, [4, 16, 20, 21, 22, 28]] = 1
        #     Lx[11, [6, 18, 20, 22, 23, 30]] = 1
        #     Lx[12, [8, 16, 24, 25, 26, 28]] = 1
        #     Lx[13, [10, 18, 24, 26, 27, 30]] = 1
        #     Lx[14, [12, 20, 24, 28, 29, 30]] = 1
        #     Lx[15, [14, 22, 26, 28, 30, 31]] = 1

        #     Lz[0, 1] = 1
        #     Lz[1, 3] = 1
        #     Lz[2, 5] = 1
        #     Lz[3, 7] = 1
        #     Lz[4, 9] = 1
        #     Lz[5, 11] = 1
        #     Lz[6, 13] = 1
        #     Lz[7, 15] = 1
        #     Lz[8, 17] = 1
        #     Lz[9, 19] = 1
        #     Lz[10, 21] = 1
        #     Lz[11, 23] = 1
        #     Lz[12, 25] = 1
        #     Lz[13, 27] = 1
        #     Lz[14, 29] = 1
        #     Lz[15, 31] = 1

        # else:
            # raise NotImplementedError("NotImplementedError")
        
        return Hz, Lx, Lz