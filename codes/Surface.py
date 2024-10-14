import numpy as np
from scipy.sparse import csr_matrix

class SurfaceCode:
    def __init__(self, L):
        self.L = L
        self.N = 2 * L ** 2 - 2 * L + 1
        self.k = 1
        self.hx = self.generate_x_stabilisers()
        self.hz = self.generate_z_stabilisers()
        self.lx = self.generate_x_logicals()
        self.lz = self.generate_z_logicals()

    def generate_x_stabilisers(self):
        """
        生成Surface Code的X稳定子矩阵
        """
        code_length = 2 * self.L ** 2 - 2 * self.L + 1
        Hx = np.zeros([int((code_length-1)/2), code_length], dtype=np.uint8)

        for i in range(self.L-1):
            for j in range(self.L):
                if j == 0:
                    Hx[i * self.L + j, i * (2 * self.L - 1) + j] = 1
                    Hx[i * self.L + j, (i + 1) * (2 * self.L - 1) + j] = 1
                    Hx[i * self.L + j, i * (2 * self.L - 1) + j + self.L] = 1
                elif j == self.L-1:
                    Hx[i * self.L + j, i * (2 * self.L - 1) + j] = 1
                    Hx[i * self.L + j, (i + 1) * (2 * self.L - 1) + j] = 1
                    Hx[i * self.L + j, i * (2 * self.L - 1) + j + self.L - 1] = 1
                else:
                    Hx[i * self.L + j, i * (2 * self.L - 1) + j] = 1
                    Hx[i * self.L + j, (i + 1) * (2 * self.L - 1) + j] = 1
                    Hx[i * self.L + j, i * (2 * self.L - 1) + j + self.L - 1] = 1
                    Hx[i * self.L + j, i * (2 * self.L - 1) + j + self.L] = 1

        return csr_matrix(Hx).toarray()

    def generate_z_stabilisers(self):
        """
        生成Surface Code的Z稳定子矩阵
        """
        code_length = 2 * self.L ** 2 - 2 * self.L + 1
        Hz = np.zeros([int((code_length-1)/2), code_length], dtype=np.uint8)

        for j in range(self.L-1):
            for i in range(self.L):
                if i == 0:
                    Hz[j * self.L + i, i * (2 * self.L - 1) + j] = 1
                    Hz[j * self.L + i, i * (2 * self.L - 1) + j + 1] = 1
                    Hz[j * self.L + i, i * (2 * self.L - 1) + j + self.L] = 1
                elif i == self.L-1:
                    Hz[j * self.L + i, i * (2 * self.L - 1) + j] = 1
                    Hz[j * self.L + i, i * (2 * self.L - 1) + j + 1] = 1
                    Hz[j * self.L + i, i * (2 * self.L - 1) + j - (self.L - 1)] = 1
                else:
                    Hz[j * self.L + i, i * (2 * self.L - 1) + j] = 1
                    Hz[j * self.L + i, i * (2 * self.L - 1) + j + 1] = 1
                    Hz[j * self.L + i, i * (2 * self.L - 1) + j + self.L] = 1
                    Hz[j * self.L + i, i * (2 * self.L - 1) + j - (self.L - 1)] = 1

        return csr_matrix(Hz).toarray()

    def generate_x_logicals(self):
        """
        生成Surface Code的X逻辑操作符矩阵
        """
        code_length = 2 * self.L ** 2 - 2 * self.L + 1
        x_logicals = np.zeros([1, code_length], dtype=np.uint8)
        for i in range(self.L):
            x_logicals[0, i] = 1

        return csr_matrix(x_logicals).toarray()

    def generate_z_logicals(self):
        """
        生成Surface Code的Z逻辑操作符矩阵
        """
        code_length = 2 * self.L ** 2 - 2 * self.L + 1
        z_logicals = np.zeros([1, code_length], dtype=np.uint8)

        for i in range(self.L):
            z_logicals[0, i * (2 * self.L - 1)] = 1

        return csr_matrix(z_logicals).toarray()
