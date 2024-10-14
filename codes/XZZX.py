import numpy as np
from scipy.sparse import csr_matrix

class XZZXCode:
    def __init__(self, L):
        """
        初始化 XZZX Code 类，生成给定 lattice 大小 L 的 X 和 Z 稳定子矩阵及逻辑操作符矩阵。
        """
        self.L = L
        self.hx = self.generate_x_stabilisers()
        self.hz = self.generate_z_stabilisers()
        self.lx = self.generate_x_logicals()
        self.lz = self.generate_z_logicals()
        self.N = self.hx.shape[1]
        self.k = 1

    def generate_x_stabilisers(self):
        """
        生成 XZZX Code 的 X 稳定子矩阵。
        """
        Hx = np.zeros([self.L**2-1, self.L**2], dtype=np.uint8)
        for i in range(self.L-1):
            for j in range(self.L-1):
                Hx[i * (self.L-1) + j, i * self.L + j] = 1
                Hx[i * (self.L-1) + j, i * self.L + j + self.L + 1] = 1
        
        k = (self.L-1) ** 2
        for i in range(self.L):
            if i == 0:
                for j in range(1, self.L):
                    if j % 2 == 0:
                        Hx[k, i * self.L + j] = 1
                        k += 1
            if i == self.L-1:
                for j in range(0, self.L-1):
                    if j % 2 == 0:
                        Hx[k, i * self.L + j] = 1
                        k += 1

        for j in range(self.L):
            if j == 0:
                for i in range(1, self.L):
                    if i % 2 == 1:
                        Hx[k, i * self.L + j] = 1
                        k += 1
            if j == self.L-1:
                for i in range(0, self.L-1):
                    if i % 2 == 1:
                        Hx[k, i * self.L + j] = 1
                        k += 1

        return csr_matrix(Hx).toarray()

    def generate_z_stabilisers(self):
        """
        生成 XZZX Code 的 Z 稳定子矩阵。
        """
        Hz = np.zeros([self.L ** 2 - 1, self.L ** 2], dtype=np.uint8)
        for i in range(self.L - 1):
            for j in range(self.L - 1):
                Hz[i * (self.L - 1) + j, i * self.L + j + 1] = 1
                Hz[i * (self.L - 1) + j, i * self.L + j + self.L] = 1
        
        k = (self.L - 1) ** 2
        for i in range(self.L):
            if i == 0:
                for j in range(1, self.L):
                    if j % 2 == 1:
                        Hz[k, i * self.L + j] = 1
                        k += 1
            if i == self.L-1:
                for j in range(0, self.L-1):
                    if j % 2 == 1:
                        Hz[k, i * self.L + j] = 1
                        k += 1

        for j in range(self.L):
            if j == 0:
                for i in range(0, self.L-1):
                    if i % 2 == 0:
                        Hz[k, i * self.L + j] = 1
                        k += 1
            if j == self.L-1:
                for i in range(1, self.L):
                    if i % 2 == 0:
                        Hz[k, i * self.L + j] = 1
                        k += 1

        return csr_matrix(Hz).toarray()

    def generate_x_logicals(self):
        """
        生成 XZZX Code 的 X 逻辑操作符矩阵。
        """
        x_logicals = np.zeros([1, 2 * self.L ** 2], dtype=np.uint8)
        for i in range(self.L):
            if i % 2 == 0:
                x_logicals[0, i * self.L] = 1
            else:
                x_logicals[0, i * self.L + self.L ** 2] = 1
        return x_logicals

    def generate_z_logicals(self):
        """
        生成 XZZX Code 的 Z 逻辑操作符矩阵。
        """
        z_logicals = np.zeros([1, 2 * self.L ** 2], dtype=np.uint8)
        for i in range(self.L):
            if i % 2 == 1:
                z_logicals[0, i] = 1
            else:
                z_logicals[0, i + self.L ** 2] = 1
        return z_logicals

    def show_stabilizers(self):
        """
        打印 X 和 Z 稳定子矩阵中的非零元素。
        """
        print("X stabilizers:")
        for r in range(self.Hx.shape[0]):
            nonZeroElementsHx = np.where(self.Hx[r, :] != 0)[0] + 1
            print(f"Row {r + 1}: {nonZeroElementsHx}")

        print("Z stabilizers:")
        for r in range(self.Hz.shape[0]):
            nonZeroElementsHz = np.where(self.Hz[r, :] != 0)[0] + 1
            print(f"Row {r + 1}: {nonZeroElementsHz}")
