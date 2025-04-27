import numpy as np
from scipy.linalg import cholesky, solve_triangular
from qpsolvers import solve_qp

def mpc(x0, target, Ts):
    """
    Model Predictive Control solver for pursuer trajectory optimization
    """
    N = 10
    # Control Input Constraint Matrices
    E = np.vstack([np.eye(2*N), -np.eye(2*N)])
    W = np.ones(4*N) * 5
    # System dynamics matrices
    m = 1     # Mass
    A = np.array([
        [1, 0, Ts, 0],
        [0, 1, 0, Ts],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    B = np.zeros((4, 2))
    B[2, 0] = Ts/m
    B[3, 1] = Ts/m
    Q = np.diag([10, 10, 1, 1])
    R = np.diag([1, 1])
    # MPC setup
    P = Q  # Terminal cost same as stage cost
    
    # Building G matrix (equivalent to MATLAB's block formation)
    G = np.zeros((N*4, N*2))
    for i in range(N):
        for j in range(N):
            if i >= j:
                G[i*4:(i+1)*4, j*2:(j+1)*2] = np.linalg.matrix_power(A, i-j) @ B
    
    # Building block diagonal matrices
    Qbar = np.zeros((N*4, N*4))
    for i in range(N-1):
        Qbar[i*4:(i+1)*4, i*4:(i+1)*4] = Q
    Qbar[(N-1)*4:N*4, (N-1)*4:N*4] = P
    
    Rbar = np.zeros((N*2, N*2))
    for i in range(N):
        Rbar[i*2:(i+1)*2, i*2:(i+1)*2] = R
    
    # Cost function matrices
    L = G.T @ Qbar @ G + Rbar
    epsilon = 1e-6
    L = L + epsilon * np.eye(L.shape[0])
    
    # Building H matrix
    H = np.zeros((N*4, 4))
    for i in range(N):
        H[i*4:(i+1)*4, :] = np.linalg.matrix_power(A, i+1)
    
    F = G.T @ Qbar @ H
    
    # Cholesky decomposition and inverse
    Lo = cholesky(L, lower=True)
    Linv = solve_triangular(Lo, np.eye(Lo.shape[0]), lower=True)
    
    # Active set solver emulation (simplified for Python conversion)
    # Note: Full active set solver implementation would be more complex
    u = solve_qp(L, F @ (x0 - target), E, W, solver="cvxopt")

    # Prediction steps
    uMPC = u[:2]  # First control input
    
    xMPC = np.zeros((4, N+1))
    xMPC[:, 0] = x0
    
    for i in range(N):
        xMPC[:, i+1] = A @ xMPC[:, i] + B @ u[i*2:(i+1)*2]
    
    return uMPC, xMPC