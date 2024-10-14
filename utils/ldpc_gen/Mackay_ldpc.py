import numpy as np

def mackay_ldpc_code(M, N, parity_degree, bit_degree, n_retries=100, backtracking_depth=2):
    assert M * parity_degree == N * bit_degree, "Incompatible degrees"
    
    # Initialize the parity-check matrix
    parity_check = np.zeros((M, N), dtype=int)
    
    def backtrack(current_col):
        s = max(0, current_col - backtracking_depth)
        # Erase the last t + 1 columns
        parity_check[:, s:current_col] = 0
        return s
    
    def check_row_degree(until):
        return np.all(parity_check[:, :until].sum(axis=1) <= parity_degree)
    
    j = 0
    while j < N:
        successful_generation = False
        for _ in range(n_retries):
            # Delete column j
            parity_check[:, j] = 0
            # Randomly generate column j with weight bit_degree
            indices = np.random.choice(M, bit_degree, replace=False)
            parity_check[indices, j] = 1
            
            if check_row_degree(j + 1):
                successful_generation = True
                break
        
        if successful_generation:
            j += 1
        else:
            j = backtrack(j)
    
    return parity_check

def test():
    # Example usage
    M, N = 6, 8  # Example matrix dimensions
    bit_degree = 3  # Example bit degree
    parity_degree = 4  # Example parity degree
    n_retries = 100  # Number of retries for each column generation
    backtracking_depth = 2  # Backtracking depth

    ldpc_matrix = mackay_ldpc_code(M, N, parity_degree, bit_degree, n_retries, backtracking_depth)
    # np.savetxt(f"output_mackay/6*8.txt", ldpc_matrix, fmt="%d")
    print(ldpc_matrix)
    print(np.linalg.matrix_rank(ldpc_matrix))

