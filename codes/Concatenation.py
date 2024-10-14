import numpy as np

class Concatenation:
    def __init__(self, Hx1=None, Hz1=None, Lx1=None, Lz1=None, Hx2=None, Hz2=None, Lx2=None, Lz2=None):
        """
        初始化 Concatenation 类，连接两层量子码的稳定子和逻辑操作符矩阵。
        """
        self.Hx1 = Hx1
        self.Hz1 = Hz1
        self.Lx1 = Lx1
        self.Lz1 = Lz1
        self.Hx2 = Hx2
        self.Hz2 = Hz2
        self.Lx2 = Lx2
        self.Lz2 = Lz2

        # 执行连接操作
        self.hx, self.hz, self.lx, self.lz = self.concatenate_codes()

    def concatenate_codes(self, Hx1, Hz1, Lx1=None, Lz1=None, Hx2=None, Hz2=None, Lx2=None, Lz2=None):
        """
        Concatenates two sets of quantum error-correcting code matrices.

        Args:
        Hx1, Hz1, Lx1, Lz1 (numpy.ndarray): Stabilizer and logical operator matrices of the first layer code.
        Hx2, Hz2, Lx2, Lz2 (numpy.ndarray): Stabilizer and logical operator matrices of the second layer code.

        Returns:
        tuple: Concatenated stabilizer and logical operator matrices (Hx, Hz, Lx, Lz).
        """

        if Hx1 is None:
            Hx1 = np.zeros_like(Hz1) 
        if Hz1 is None:
            Hz1 = np.zeros_like(Hx1)
        
        if Hx2 is None and Hz2 is None:
            Hx2 = Hz1
            Hz2 = Hx1
            Lx2 = Lz1
            Lz2 = Lx1
        elif Hx2 is None:
            Hx2 = np.zeros_like(Hz2)
        elif Hz2 is None:
            Hz2 = np.zeros_like(Hx2)

        k1 = Lx1.shape[0]
        code_length = Hx1.shape[1] * Hx2.shape[1] // k1

        k = Lx2.shape[0]
        n_block = Hx2.shape[1] // k1  # Number of blocks in the first layer code
        Hx = np.zeros((n_block * Hx1.shape[0] + Hx2.shape[0], code_length))
        Hz = np.zeros((n_block * Hz1.shape[0] + Hz2.shape[0], code_length))

        # First layer stabilizers
        for i in range(n_block):
            Hx[i * Hx1.shape[0]:(i + 1) * Hx1.shape[0], i * Hx1.shape[1]:(i + 1) * Hx1.shape[1]] = Hx1
            Hz[i * Hz1.shape[0]:(i + 1) * Hz1.shape[0], i * Hz1.shape[1]:(i + 1) * Hz1.shape[1]] = Hz1
        
        # Second layer stabilizers
        Hx[n_block * Hx1.shape[0]:, :] = self.process_matrix(Hx2, Lx1, k1, n_block)
        Hz[n_block * Hz1.shape[0]:, :] = self.process_matrix(Hz2, Lz1, k1, n_block)

        # Logical operators
        if np.all(Lx1 == 0):
            Lx = np.zeros(Lx1.shape)
            Lz = np.zeros(Lz1.shape)
        else:
            Lx = self.process_matrix(Lx2, Lx1, k1, n_block)
            Lz = self.process_matrix(Lz2, Lz1, k1, n_block)

        return Hx, Hz, Lx, Lz


    def process_matrix(Hx2, Lx1, k, nGroups):
        nRows = Hx2.shape[0]  # Get the number of rows of Hx2

        # Initialize the result matrix
        result = np.zeros((nRows, nGroups * Lx1.shape[1]), dtype=int)

        for row in range(nRows):
            for group in range(nGroups):
                groupVector = np.zeros(Lx1.shape[1], dtype=int)  # Initialize group vector
                groupStart = group * k
                groupEnd = (group + 1) * k

                # Iterate through each bit in the group
                for bit in range(groupStart, groupEnd):
                    if Hx2[row, bit] == 1:
                        bitIndex = bit % Lx1.shape[0]  # Indexing Lx1 using modulo
                        groupVector = (groupVector + Lx1[bitIndex, :]) % 2  # Modulo 2 addition

                # Add the group vector to the result matrix
                result[row, group * Lx1.shape[1]:(group + 1) * Lx1.shape[1]] = groupVector

        return result