import numpy as np

def commutation_check(matrix, v, n):
    return not np.any(((matrix[:, :n] @ v[:, n:].T) + (matrix[:, n:] @ v[:, :n].T))%2)

def rank(matrix):
    r = 0
    col = 0

    while True:
        row_index, col_index = find_left_up_most_index(matrix[r:, col:])
        if col_index < 0:
            break
        col_index += col
        row_index += r
        # print("col = {}".format(col))
        # print("row_index = {}".format(row_index))
        # print("col_index = {}".format(col_index))
        # print("matrix = {}".format(matrix))
        col = col_index + 1

        for j in range(0, r):
            if matrix[j, col_index] == 1:
                matrix[j] = (matrix[j] + matrix[row_index]) % 2

        for j in range(row_index + 1, matrix.shape[0]):
            if matrix[j, col_index] == 1:
                matrix[j] = (matrix[j] + matrix[row_index]) % 2

        row = np.tile(matrix[r], 1)
        matrix[r] = matrix[row_index]
        matrix[row_index] = row

        column = np.tile(matrix[:, r], 1)
        matrix[:, r] = matrix[:, col_index]
        matrix[:, col_index] = column
        r += 1

    return r

def independent_check(Matrix, vector=np.matrix([[]])):
    matrix = np.tile(Matrix, 1)
    r1 = rank(matrix)
    if vector.shape[1] == 0:
        return r1 == matrix.shape[0]
    return r1 + vector.shape[0] == rank(np.append(matrix, vector, axis=0))

def find_left_up_most_index(matrix):
    for j in range(0, matrix.shape[1]):
        for i in range(0, matrix.shape[0]):
            if matrix[i, j] == 1:
                return i, j
    return -1, -1

def non_CSS_logical_operator(Matrix):
    matrix = np.tile(Matrix, 1)
    if not independent_check(matrix):
        print("not independent check matrix")
    matrix = np.tile(Matrix, 1)
    n = int(matrix.shape[1] / 2)
    r_left = 0
    r_right = 0

    column_swap_record = np.zeros([1, n], dtype=np.uint16)
    for i in range(0, n):
        column_swap_record[0, i] = i + 1

    i = 0
    col = 0
    while True:
        row_index, col_index = find_left_up_most_index(matrix[i:, col:])
        col_index += col
        row_index += i
        # print("col = {}".format(col))
        # print("row_index = {}".format(row_index))
        # print("col_index = {}".format(col_index))
        # print("matrix = {}".format(matrix))
        if col_index >= n or col_index < col:
            break

        col = col_index + 1

        for j in range(0, i):
            if matrix[j, col_index] == 1:
                matrix[j] = (matrix[j] + matrix[row_index]) % 2

        for j in range(row_index + 1, matrix.shape[0]):
            if matrix[j, col_index] == 1:
                matrix[j] = (matrix[j] + matrix[row_index]) % 2

        row = np.tile(matrix[i], 1)
        matrix[i] = matrix[row_index]
        matrix[row_index] = row

        column = np.tile(matrix[:, i], 1)
        matrix[:, i] = matrix[:, col_index]
        matrix[:, col_index] = column

        column = np.tile(matrix[:, i + n], 1)
        # print("i = {}".format(i))
        # print("r = {}".format(r))

        matrix[:, i + n] = matrix[:, col_index + n]
        matrix[:, col_index + n] = column

        qubit_index = column_swap_record[0, i]
        column_swap_record[0, i] = column_swap_record[0, col_index]
        column_swap_record[0, col_index] = qubit_index

        i += 1

    r_left = i
    # print(matrix)
    # print(column_swap_record)

    full_r = False
    E = np.zeros([1, n - matrix.shape[0]], dtype=np.uint16)
    if not r_left == matrix.shape[0]:

        i = 0
        col = n + r_left
        while True:
            row_index, col_index = find_left_up_most_index(matrix[r_left + i:, col:])
            if col_index < 0:
                break
            col_index += col
            row_index += r_left + i
            # print(row_index)
            # print(col_index)
            # print(i + r)
            # print(col)
            # print(matrix)

            for j in range(row_index + 1, matrix.shape[0]):
                if matrix[j, col_index] == 1:
                    matrix[j] = (matrix[j] + matrix[row_index]) % 2

            for j in range(0, r_left + i):
                if matrix[j, col_index] == 1:
                    matrix[j] = (matrix[j] + matrix[row_index]) % 2

            row = np.tile(matrix[i + r_left], 1)
            matrix[i + r_left] = matrix[row_index]
            matrix[row_index] = row
            # print(matrix)
            column = np.tile(matrix[:, i + n + r_left], 1)
            matrix[:, i + n + r_left] = matrix[:, col_index]
            matrix[:, col_index] = column

            column = np.tile(matrix[:, i + r_left], 1)
            matrix[:, i + r_left] = matrix[:, col_index - n]
            matrix[:, col_index - n] = column

            qubit_index = column_swap_record[0, i + r_left]
            column_swap_record[0, i + r_left] = column_swap_record[0, col_index - n]
            column_swap_record[0, col_index - n] = qubit_index

            i += 1

        r_right = i
        E = matrix[r_left:r_left + r_right, r_left + r_right + n:]
    else:
        full_r = True
    # print("matrix = ")
    # print(matrix)
    # print("r_left = {}".format(r_left))
    # print("r_right = {}".format(r_right))
    # print(column_swap_record)
    A2 = matrix[:r_left, r_left + r_right:n]
    C = matrix[:r_left, r_left + r_right + n:]

    k = n - r_left - r_right
    # print("k = {}".format(k))
    
    I = np.zeros([1, k], dtype=np.uint16)
    for i in range(0, k):
        new_row = np.zeros([1, k], dtype=np.uint16)
        new_row[0, i] = 1
        I = np.append(I, new_row, axis=0)
    I = I[1:]

    zero_column = np.zeros([1, k], dtype=np.uint16)

    L_Z_matrix = np.zeros([1, k], dtype=np.uint16).T
    for i in range(0, n):
        L_Z_matrix = np.append(L_Z_matrix, zero_column.T, axis=1)
    L_Z_matrix = np.append(L_Z_matrix, A2.T, axis=1)

    for i in range(0, r_right):
        L_Z_matrix = np.append(L_Z_matrix, zero_column.T, axis=1)
    L_Z_matrix = np.append(L_Z_matrix, I, axis=1)
    L_Z_matrix = L_Z_matrix[:, 1:]

    L_X_matrix = np.zeros([1, k], dtype=np.uint16).T
    for i in range(0, r_left):
        L_X_matrix = np.append(L_X_matrix, zero_column.T, axis=1)
    if not full_r:
        L_X_matrix = np.append(L_X_matrix, E.T, axis=1)
    L_X_matrix = np.append(L_X_matrix, I, axis=1)
    L_X_matrix = np.append(L_X_matrix, C.T, axis=1)
    # print("---------------------------------------------")
    # print(A2.T)
    # print(E.T)
    # print(C.T)
    # print(I)
    for i in range(0, n - r_left):
        L_X_matrix = np.append(L_X_matrix, zero_column.T, axis=1)
    L_X_matrix = L_X_matrix[:, 1:]

    lx = np.zeros([1, 2*n], dtype=np.uint16)
    for i in range(0, L_X_matrix.shape[0]):
        new_row = np.zeros([1, 2*n], dtype=np.uint16)
        for j in range(0, 2*n):
            if L_X_matrix[i, j] == 1:
                new_row[0, column_swap_record[0, j % n] - 1 + int(j/n) * n] = 1
        lx = np.append(lx, new_row, axis=0)
    lx = lx[1:]

    lz = np.zeros([1, 2 * n], dtype=np.uint16)
    for i in range(0, L_Z_matrix.shape[0]):
        new_row = np.zeros([1, 2 * n], dtype=np.uint16)
        for j in range(0, 2 * n):
            if L_Z_matrix[i, j] == 1:
                new_row[0, column_swap_record[0, j % n] - 1 + int(j / n) * n] = 1
        lz = np.append(lz, new_row, axis=0)
    lz = lz[1:]
    if not commutation_check(Matrix, lx, n):
        print("logical x operator not commute with check matrix")

    if not independent_check(Matrix, lx):
        print("logical x operator not logical")

    if not commutation_check(Matrix, lz, n):
        print("logical z operator not commute with check matrix")

    if not independent_check(Matrix, lz):
        print("logical z operator not logical")
    # print("Lx = ")
    # print(L_X_matrix)
    # print("Lz = {}")
    # print(L_Z_matrix)
    return lx, lz


def css_logical_operator(Hx, Hz):
    Hx0 = np.zeros(Hx.shape).astype(np.uint8)
    Hz0 = np.zeros(Hz.shape).astype(np.uint8)
    Hx_bar = np.vstack([Hx, Hx0])
    Hz_bar = np.vstack([Hz0, Hz])
    Matrix = np.hstack([Hx_bar, Hz_bar])
    matrix = np.tile(Matrix, 1)
    if not independent_check(matrix):
        print("not independent check matrix")
    matrix = np.tile(Matrix, 1)
    n = int(matrix.shape[1] / 2)
    r_left = 0
    r_right = 0

    column_swap_record = np.zeros([1, n], dtype=np.uint16)
    for i in range(0, n):
        column_swap_record[0, i] = i + 1

    i = 0
    col = 0
    while True:
        row_index, col_index = find_left_up_most_index(matrix[i:, col:])
        col_index += col
        row_index += i
        # print("col = {}".format(col))
        # print("row_index = {}".format(row_index))
        # print("col_index = {}".format(col_index))
        # print("matrix = {}".format(matrix))
        if col_index >= n or col_index < col:
            break

        col = col_index + 1

        for j in range(0, i):
            if matrix[j, col_index] == 1:
                matrix[j] = (matrix[j] + matrix[row_index]) % 2

        for j in range(row_index + 1, matrix.shape[0]):
            if matrix[j, col_index] == 1:
                matrix[j] = (matrix[j] + matrix[row_index]) % 2

        row = np.tile(matrix[i], 1)
        matrix[i] = matrix[row_index]
        matrix[row_index] = row

        column = np.tile(matrix[:, i], 1)
        matrix[:, i] = matrix[:, col_index]
        matrix[:, col_index] = column

        column = np.tile(matrix[:, i + n], 1)
        # print("i = {}".format(i))
        # print("r = {}".format(r))

        matrix[:, i + n] = matrix[:, col_index + n]
        matrix[:, col_index + n] = column

        qubit_index = column_swap_record[0, i]
        column_swap_record[0, i] = column_swap_record[0, col_index]
        column_swap_record[0, col_index] = qubit_index

        i += 1

    r_left = i
    # print(matrix)
    # print(column_swap_record)

    full_r = False
    E = np.zeros([1, n - matrix.shape[0]], dtype=np.uint16)
    if not r_left == matrix.shape[0]:

        i = 0
        col = n + r_left
        while True:
            row_index, col_index = find_left_up_most_index(matrix[r_left + i:, col:])
            if col_index < 0:
                break
            col_index += col
            row_index += r_left + i
            # print(row_index)
            # print(col_index)
            # print(i + r)
            # print(col)
            # print(matrix)

            for j in range(row_index + 1, matrix.shape[0]):
                if matrix[j, col_index] == 1:
                    matrix[j] = (matrix[j] + matrix[row_index]) % 2

            for j in range(0, r_left + i):
                if matrix[j, col_index] == 1:
                    matrix[j] = (matrix[j] + matrix[row_index]) % 2

            row = np.tile(matrix[i + r_left], 1)
            matrix[i + r_left] = matrix[row_index]
            matrix[row_index] = row
            # print(matrix)
            column = np.tile(matrix[:, i + n + r_left], 1)
            matrix[:, i + n + r_left] = matrix[:, col_index]
            matrix[:, col_index] = column

            column = np.tile(matrix[:, i + r_left], 1)
            matrix[:, i + r_left] = matrix[:, col_index - n]
            matrix[:, col_index - n] = column

            qubit_index = column_swap_record[0, i + r_left]
            column_swap_record[0, i + r_left] = column_swap_record[0, col_index - n]
            column_swap_record[0, col_index - n] = qubit_index

            i += 1

        r_right = i
        E = matrix[r_left:r_left + r_right, r_left + r_right + n:]
    else:
        full_r = True
    # print("matrix = ")
    # print(matrix)
    # print("r_left = {}".format(r_left))
    # print("r_right = {}".format(r_right))
    # print(column_swap_record)
    A2 = matrix[:r_left, r_left + r_right:n]
    C = matrix[:r_left, r_left + r_right + n:]

    k = n - r_left - r_right
    # print("k = {}".format(k))
    
    I = np.zeros([1, k], dtype=np.uint16)
    for i in range(0, k):
        new_row = np.zeros([1, k], dtype=np.uint16)
        new_row[0, i] = 1
        I = np.append(I, new_row, axis=0)
    I = I[1:]

    zero_column = np.zeros([1, k], dtype=np.uint16)

    L_Z_matrix = np.zeros([1, k], dtype=np.uint16).T
    for i in range(0, n):
        L_Z_matrix = np.append(L_Z_matrix, zero_column.T, axis=1)
    L_Z_matrix = np.append(L_Z_matrix, A2.T, axis=1)

    for i in range(0, r_right):
        L_Z_matrix = np.append(L_Z_matrix, zero_column.T, axis=1)
    L_Z_matrix = np.append(L_Z_matrix, I, axis=1)
    L_Z_matrix = L_Z_matrix[:, 1:]

    L_X_matrix = np.zeros([1, k], dtype=np.uint16).T
    for i in range(0, r_left):
        L_X_matrix = np.append(L_X_matrix, zero_column.T, axis=1)
    if not full_r:
        L_X_matrix = np.append(L_X_matrix, E.T, axis=1)
    L_X_matrix = np.append(L_X_matrix, I, axis=1)
    L_X_matrix = np.append(L_X_matrix, C.T, axis=1)
    # print("---------------------------------------------")
    # print(A2.T)
    # print(E.T)
    # print(C.T)
    # print(I)
    for i in range(0, n - r_left):
        L_X_matrix = np.append(L_X_matrix, zero_column.T, axis=1)
    L_X_matrix = L_X_matrix[:, 1:]

    lx = np.zeros([1, 2*n], dtype=np.uint16)
    for i in range(0, L_X_matrix.shape[0]):
        new_row = np.zeros([1, 2*n], dtype=np.uint16)
        for j in range(0, 2*n):
            if L_X_matrix[i, j] == 1:
                new_row[0, column_swap_record[0, j % n] - 1 + int(j/n) * n] = 1
        lx = np.append(lx, new_row, axis=0)
    lx = lx[1:]

    lz = np.zeros([1, 2 * n], dtype=np.uint16)
    for i in range(0, L_Z_matrix.shape[0]):
        new_row = np.zeros([1, 2 * n], dtype=np.uint16)
        for j in range(0, 2 * n):
            if L_Z_matrix[i, j] == 1:
                new_row[0, column_swap_record[0, j % n] - 1 + int(j / n) * n] = 1
        lz = np.append(lz, new_row, axis=0)
    lz = lz[1:]
    if not commutation_check(Matrix, lx, n):
        print("logical x operator not commute with check matrix")

    if not independent_check(Matrix, lx):
        print("logical x operator not logical")

    if not commutation_check(Matrix, lz, n):
        print("logical z operator not commute with check matrix")

    if not independent_check(Matrix, lz):
        print("logical z operator not logical")
    # print("Lx = ")
    # print(L_X_matrix)
    # print("Lz = {}")
    # print(L_Z_matrix)
    return lx[:, :Hx.shape[1]], lz[:, Hz.shape[1]:]

# matrix = np.matrix([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#                     [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
#                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]])
# lx, lz = stabiliser_standardization(matrix)
# print(lx)
# print(lz)

