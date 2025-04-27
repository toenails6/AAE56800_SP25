import numpy as np
from scipy.linalg import cholesky, solve_triangular
from qpsolvers import solve_qp

def mpc(x_est_evader, x_est_pursuer, maxSpeed, areaBnds, Ts):
    
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
    P = Q  # Terminal cost same as stage cost

    def compute_optimal_targets(x_est_evader, x_est_pursuer, N, Ts):
        # For evader: compute optimal target that maximizes distance from pursuer
        # Compute a safe direction away from pursuer
        direction = x_est_evader[0:2] - x_est_pursuer[0:2]
        norm = np.linalg.norm(direction)
        direction = direction / norm  # compute cos, sin

        # Calculate target position
        target_x = x_est_evader[0] + direction[0] * 20  # Look ahead
        target_y = x_est_evader[1] + direction[1] * 20
        optimal_target_evader = np.array([target_x, target_y, x_est_evader[2], x_est_evader[3]])
        
        # For pursuer: compute optimal target that minimizes distance to evader
        # Predict evader's future position for interception
        predicted_evader_pos = x_est_evader[0:2] + x_est_evader[2:4] * N * Ts
        
        # Set pursuer's target to intercept the evader's predicted position
        optimal_target_pursuer = np.array([
            predicted_evader_pos[0], 
            predicted_evader_pos[1], 
            x_est_evader[2],  # Match evader's velocity for interception
            x_est_evader[3]
        ])
        return optimal_target_evader, optimal_target_pursuer
    target, optimal_target_pursuer = compute_optimal_targets(x_est_evader, x_est_pursuer, N, Ts)

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
    u = solve_qp(L, F @ (x_est_evader - target), E, W, solver="cvxopt")

    # Prediction steps
    uMPC = u[:2]  # First control input
    
    xMPC = np.zeros((4, N+1))
    xMPC[:, 0] = x_est_evader
    
    ## limit velcoity
    for i in range(N):
        next_evader_state = A @ x_est_evader + B @ uMPC
        evader_speed = np.sqrt(next_evader_state[2]**2 + next_evader_state[3]**2)
        if evader_speed > maxSpeed:
            next_evader_state[2] = maxSpeed
            next_evader_state[3] = maxSpeed
        xMPC[:, i+1] = next_evader_state

    return xMPC[:, 1][0], xMPC[:, 1][1], 2