import numpy as np
import math

class XZTGRE:
    def __init__(self, k):
        self.hx, self.hz = self.H_matrix(k)
        self.hx = self.hx[~np.all(self.hx == 0, axis=1)]
        self.hz = self.hz[~np.all(self.hz == 0, axis=1)]
        self.lx, self.lz = self.logical_operator(k)
        self.N = self.hx.shape[1]
        self.k = self.N - self.hx.shape[0] - self.hz.shape[0]

    def CSS_check_matrix_Zero_PAD(self, Hx, Hz):
        n1 = Hx.shape[0]
        n2 = Hz.shape[0]
        code_length_N = Hx.shape[1]
        if n1 > 0:
            for i in range(0, n1):
                Indentity_row = np.zeros([1, code_length_N], dtype=np.uint16)
                Hx = np.append(Hx, Indentity_row, axis=0)

        zeromatrix = np.zeros([1, code_length_N], dtype=np.uint16)
        if n2 > 0:
            for i in range(0, n2 - 1):
                Indentity_row = np.zeros([1, code_length_N], dtype=np.uint16)
                zeromatrix = np.append(zeromatrix, Indentity_row, axis=0)
        # print("zeromatrix = ")
        # print(zeromatrix)
        # print("hz")
        # print(Hz)
        Hz = np.append(zeromatrix, Hz, axis=0)
        return Hx, Hz

    def TGRE_StabGene_SubscriptMatrix_generate(self, k):
        matrix = np.zeros([1, 1], dtype=np.uint16)
        matrix[0, 0] = 1
        for i in range(0, k - 1):
            a = 2 ** (i + 1) + 1
            # print("matrix")
            # print(matrix)
            new_matrix_u = np.tile(matrix, 1)
            new_column_r = np.zeros([1, 1], dtype=np.uint16)
            for j in range(0, new_matrix_u.shape[0]):
                new_subscript = np.zeros([1, 1], dtype=np.uint16)
                new_subscript[0, 0] = j * 2 + a
                new_column_r = np.append(new_column_r, new_subscript, axis=1)
            new_column_r = new_column_r.reshape(-1, 1)
            new_column_r = new_column_r[1:]
            # print("column_r=")
            # print(new_column_r)
            new_matrix_u = np.append(new_matrix_u, new_column_r, axis=1)

            new_matrix_d = np.tile(matrix, 1)
            for j in range(0, new_matrix_d.shape[0]):
                for k in range(0, new_matrix_d.shape[1]):
                    new_matrix_d[j, k] += a - 1
            new_column_l = np.zeros([1, 1], dtype=np.uint16)
            for j in range(0, matrix.shape[0]):
                new_subscript = np.zeros([1, 1], dtype=np.uint16)
                new_subscript[0, 0] = j * 2 + 1
                new_column_l = np.append(new_column_l, new_subscript, axis=1)
            new_column_l = new_column_l.reshape(-1, 1)
            new_column_l = new_column_l[1:]
            new_matrix_d = np.append(new_column_l, new_matrix_d, axis=1)

            matrix = np.append(new_matrix_u, new_matrix_d, axis=0)
        # array_shift = array[(length - number_of_bits_to_shift):] + array[0:(length - number_of_bits_to_shift)]
        return matrix

    def H_matrix(self, k):
        weight_off = 1
        if k > 3:
            weight_off = int(math.log2(k - 2))

        L_weight = k + weight_off
        R_weight = k
        code_length_N = 2 ** L_weight + 2 ** (R_weight - 1)
        offset = 2**(R_weight - 2)
        subscript_matrix = self.TGRE_StabGene_SubscriptMatrix_generate(L_weight)
        subscript_matrix_z1 = np.zeros([1, L_weight], dtype=np.uint16)
        subscript_matrix_x1 = np.zeros([1, L_weight], dtype=np.uint16)

        for i in range(0, subscript_matrix.shape[0]):
            if i % 2 == 0:
                subscript_matrix_z1 = np.append(subscript_matrix_z1, subscript_matrix[i:i + 1], axis=0)
            else:
                subscript_matrix_x1 = np.append(subscript_matrix_x1, subscript_matrix[i:i + 1], axis=0)
        subscript_matrix_z1 = subscript_matrix_z1[1:]
        subscript_matrix_x1 = subscript_matrix_x1[1:]

        subscript_matrix_z_r = np.tile(subscript_matrix_z1[:offset, :R_weight], 1)
        subscript_matrix_x_r = np.tile(subscript_matrix_x1[:offset, :R_weight], 1)
        # print(subscript_matrix_z1)
        # print(subscript_matrix_z_r)
        for i in range(0, subscript_matrix_z_r.shape[0]):
            for j in range(0, subscript_matrix_z_r.shape[1]):
                subscript_matrix_z_r[i, j] += 2 ** (L_weight + 1)
                subscript_matrix_x_r[i, j] += 2 ** (L_weight + 1)

        copy = np.tile(subscript_matrix_z_r, 1)
        for i in range(0, int(pow(2, weight_off + 1) - 1)):
            subscript_matrix_z_r = np.append(subscript_matrix_z_r, copy, axis=0)

        copy = np.tile(subscript_matrix_x_r, 1)
        for i in range(0, int(pow(2, weight_off + 1) - 1)):
            subscript_matrix_x_r = np.append(subscript_matrix_x_r, copy, axis=0)

        subscript_matrix_z1_c = np.tile(subscript_matrix_z1, 1)
        for i in range(0, subscript_matrix_z1_c.shape[0]):
            for j in range(0, subscript_matrix_z1_c.shape[1]):
                subscript_matrix_z1_c[i, j] += 2 ** (L_weight)

        subscript_matrix_x1_c = np.tile(subscript_matrix_x1, 1)
        for i in range(0, subscript_matrix_x1_c.shape[0]):
            for j in range(0, subscript_matrix_x1_c.shape[1]):
                subscript_matrix_x1_c[i, j] += 2 ** (L_weight)
        subscript_matrix_z1 = np.append(subscript_matrix_z1, subscript_matrix_z1_c, axis=0)
        subscript_matrix_x1 = np.append(subscript_matrix_x1, subscript_matrix_x1_c, axis=0)

        subscript_matrix_z = np.append(subscript_matrix_z1, subscript_matrix_z_r, axis=1)
        subscript_matrix_x = np.append(subscript_matrix_x1, subscript_matrix_x_r, axis=1)

        Hx = np.zeros([1, code_length_N], dtype=np.uint16)
        for i in range(0, subscript_matrix_x.shape[0]):
            new_row = np.zeros([1, code_length_N], dtype=np.uint16)
            for j in range(0, subscript_matrix_x.shape[1]):
                new_row[0, int((subscript_matrix_x[i, j] - 1)/2)] = 1
            Hx = np.append(Hx, new_row, axis=0)

        Hz = np.zeros([1, code_length_N], dtype=np.uint16)
        for i in range(0, subscript_matrix_z.shape[0]):
            new_row = np.zeros([1, code_length_N], dtype=np.uint16)
            for j in range(0, subscript_matrix_z.shape[1]):
                new_row[0, int((subscript_matrix_z[i, j] - 1)/2)] = 1
            Hz = np.append(Hz, new_row, axis=0)
        # stabiliser_code.TGRE_matrix_presentation_print(Hz, 2**L_weight, 0)
        return self.CSS_check_matrix_Zero_PAD(Hx[1:], Hz[1:])

    def logical_operator(self, k):
        weight_off = 1
        if k > 3:
            weight_off = int(math.log2(k - 2))

        L_weight = k + weight_off
        R_weight = k
        code_length_N = 2 ** L_weight + 2 ** (R_weight - 1)
        offset = 2 ** (R_weight - 2)
        subscript_matrix = self.TGRE_StabGene_SubscriptMatrix_generate(L_weight)
        subscript_matrix_z1 = np.zeros([1, L_weight], dtype=np.uint16)
        subscript_matrix_x1 = np.zeros([1, L_weight], dtype=np.uint16)

        for i in range(0, subscript_matrix.shape[0]):
            if i % 2 == 0:
                subscript_matrix_z1 = np.append(subscript_matrix_z1, subscript_matrix[i:i + 1], axis=0)
            else:
                subscript_matrix_x1 = np.append(subscript_matrix_x1, subscript_matrix[i:i + 1], axis=0)
        subscript_matrix_z1 = subscript_matrix_z1[1:]
        subscript_matrix_x1 = subscript_matrix_x1[1:]

        subscript_matrix_z_r = np.tile(subscript_matrix_z1[:offset, :R_weight], 1)
        subscript_matrix_x_r = np.tile(subscript_matrix_x1[:offset, :R_weight], 1)
        for i in range(0, subscript_matrix_z_r.shape[0]):
            for j in range(0, subscript_matrix_z_r.shape[1]):
                subscript_matrix_z_r[i, j] += 2 ** (L_weight + 1)
                subscript_matrix_x_r[i, j] += 2 ** (L_weight + 1)

        L_X_x_matrix = np.zeros([1, code_length_N], dtype=np.uint16)
        L_X_z_matrix = np.zeros([1, code_length_N], dtype=np.uint16)
        L_Z_x_matrix = np.zeros([1, code_length_N], dtype=np.uint16)
        L_Z_z_matrix = np.zeros([1, code_length_N], dtype=np.uint16)
        zero_row = np.zeros([1, code_length_N], dtype=np.uint16)

        for i in range(0, subscript_matrix_z_r.shape[0]):
            new_row = np.zeros([1, code_length_N], dtype=np.uint16)
            for j in range(0, subscript_matrix_z_r.shape[1]):
                new_row[0, int((subscript_matrix_z_r[i, j] - 1)/2)] = 1
            L_Z_z_matrix = np.append(L_Z_z_matrix, new_row, axis=0)
            L_Z_x_matrix = np.append(L_Z_x_matrix, zero_row, axis=0)

        for i in range(0, subscript_matrix_x_r.shape[0]):
            new_row = np.zeros([1, code_length_N], dtype=np.uint16)
            for j in range(0, subscript_matrix_x_r.shape[1]):
                new_row[0, int((subscript_matrix_x_r[i, j] - 1)/2)] = 1
            L_X_x_matrix = np.append(L_X_x_matrix, new_row, axis=0)
            L_X_z_matrix = np.append(L_X_z_matrix, zero_row, axis=0)

        auxiliary_matrix_1 = np.zeros([1, code_length_N], dtype=np.uint16)
        for i in range(0, offset):
            new_row = np.zeros([1, code_length_N], dtype=np.uint16)
            new_row[0, i * 2 + 2**L_weight] = 1
            for j in range(0, 2**(weight_off + 1)):
                new_row[0, i * 2 + offset * 2 * j] = 1
            L_Z_x_matrix = np.append(L_Z_x_matrix, zero_row, axis=0)

            auxiliary_matrix_1 = np.append(auxiliary_matrix_1, new_row, axis=0)

        L_Z_z_matrix = np.append(auxiliary_matrix_1[1:], L_Z_z_matrix[1:], axis=0)

        auxiliary_matrix_2 = np.zeros([1, code_length_N], dtype=np.uint16)
        for i in range(0, offset):
            new_row = np.zeros([1, code_length_N], dtype=np.uint16)
            new_row[0, i * 2 + 2 ** L_weight + 1] = 1
            for j in range(0, 2 ** (weight_off + 1)):
                new_row[0, i * 2 + 1 + offset * 2 * j] = 1
            L_X_z_matrix = np.append(L_X_z_matrix, zero_row, axis=0)
            auxiliary_matrix_2 = np.append(auxiliary_matrix_2, new_row, axis=0)

        L_X_x_matrix = np.append(L_X_x_matrix[1:], auxiliary_matrix_2[1:], axis=0)

        return L_X_x_matrix, L_Z_z_matrix