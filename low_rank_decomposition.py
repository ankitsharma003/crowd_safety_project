# Step 4: Low-Rank Temporal Decomposition (PCA from scratch)
# This script performs PCA on the spatiotemporal matrix and computes reconstruction errors.

import pickle
import math

# Matrix utilities

def mean_center(matrix):
    n = len(matrix)
    m = len(matrix[0])
    means = [sum(matrix[i][j] for i in range(n)) / n for j in range(m)]
    centered = [[matrix[i][j] - means[j] for j in range(m)] for i in range(n)]
    return centered, means

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def matmul(A, B):
    # A: n x m, B: m x k
    n, m = len(A), len(A[0])
    m2, k = len(B), len(B[0])
    assert m == m2
    result = [[0.0 for _ in range(k)] for _ in range(n)]
    for i in range(n):
        for j in range(k):
            for l in range(m):
                result[i][j] += A[i][l] * B[l][j]
    return result

def eigen_decompose(matrix):
    # Power iteration for largest eigenvector (for simplicity)
    n = len(matrix)
    v = [1.0 for _ in range(n)]
    for _ in range(100):
        v_new = [sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x*x for x in v_new))
        v_new = [x / norm for x in v_new]
        v = v_new
    # Rayleigh quotient for eigenvalue
    eigenvalue = sum(v[i] * sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n))
    return v, eigenvalue

if __name__ == "__main__":
    with open("spatiotemporal_matrix.pkl", "rb") as f:
        matrix = pickle.load(f)
    centered, means = mean_center(matrix)
    # Covariance matrix
    X = centered
    X_T = transpose(X)
    cov = matmul(X_T, X)
    # Find first principal component
    pc1, eigval1 = eigen_decompose(cov)
    # Project data onto pc1
    projections = [sum(x[j] * pc1[j] for j in range(len(pc1))) for x in X]
    # Reconstruct from pc1
    reconstructions = [[means[j] + projections[i] * pc1[j] for j in range(len(pc1))] for i in range(len(projections))]
    # Compute reconstruction error for each frame
    errors = [math.sqrt(sum((matrix[i][j] - reconstructions[i][j])**2 for j in range(len(pc1)))) for i in range(len(matrix))]
    # Save errors and reconstructions
    with open("pca_results.pkl", "wb") as f:
        pickle.dump({
            'errors': errors,
            'reconstructions': reconstructions,
            'pc1': pc1,
            'means': means
        }, f)
    print("PCA complete. Reconstruction errors and results saved.")
