import numpy as np
from numpy.random import shuffle

def gallager_ldpc_code(n, col_weight, row_weight):
    """
    This function constructs a LDPC parity check matrix
    H. The algorithm follows Gallager's approach where we create
    p submatrices and stack them together. Reference: Turbo
    Coding for Satellite and Wireless Communications, section
    9,3.

    Note: the matrices computed from this algorithm will never
    have full rank. (Reference Gallager's Dissertation.) They
    will have rank = (number of rows - p + 1). To convert it
    to full rank, use the function get_full_rank_H_matrix
    """

    p = col_weight
    q = row_weight

    # For this algorithm, n/p must be an integer, because the
    # number of rows in each submatrix must be a whole number.
    ratioTest = (n * 1.0) / q
    if ratioTest % 1 != 0:
        print('\nError in regular_LDPC_code_contructor: The ', end='')
        print('ratio of inputs n/q must be a whole number.\n')
        return

    # First submatrix first:
    m = (n * p) // q  # number of rows in H matrix
    submatrix1 = np.zeros((m // p, n))
    for row in np.arange(m // p):
        range1 = row * q
        range2 = (row + 1) * q
        submatrix1[row, range1:range2] = 1
        H = submatrix1

    # Create the other submatrices and vertically stack them on.
    submatrixNum = 2
    newColumnOrder = np.arange(n)
    while submatrixNum <= p:
        submatrix = np.zeros((m // p, n))
        shuffle(newColumnOrder)

        for columnNum in np.arange(n):
            submatrix[:, columnNum] = \
                submatrix1[:, newColumnOrder[columnNum]]

        H = np.vstack((H, submatrix))
        submatrixNum = submatrixNum + 1

    # Double check the row weight and column weights.
    size = H.shape
    rows = size[0]
    cols = size[1]

    # Check the row weights.
    for rowNum in np.arange(rows):
        nonzeros = np.array(H[rowNum, :].nonzero())
        if nonzeros.shape[1] != q:
            print('Row', rowNum, 'has incorrect weight!')
            return

    # Check the column weights
    for columnNum in np.arange(cols):
        nonzeros = np.array(H[:, columnNum].nonzero())
        if nonzeros.shape[1] != p:
            print('Row', columnNum, 'has incorrect weight!')
            return
    return H
