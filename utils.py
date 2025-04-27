from numpy.typing import NDArray
from scipy.integrate import solve_bvp
from scipy.linalg import cholesky, solve_triangular
from qpsolvers import solve_qp
import numpy as np
import matplotlib.pyplot as plt

def chaserTPBVP(
        x_c_0: NDArray,
        x_e_0: NDArray,
        maxSpeed: float,
        areaBnds: NDArray,
        timeStep: float):
    """
    Chaser TPBVP function.

    Parameters:
    x_c_0 : array-like, shape (2,)
        Chaser initial states [x_c, y_c].
    x_e_0 : array-like, shape (4,)
        Evader initial states [x_e, y_e, v_x_e, v_y_e].
    maxSpeed : float
        Maximum speed.
    areaBnds : array-like, shape (4,)
        Area bounds [x_l, x_u, y_l, y_u].
    timeStep : float
        Time span for the simulation.

    Returns:
    ndarray, shape (2,)
        Final chaser position [x, y].
    """
    # Convert inputs to numpy arrays
    x_c_0 = np.array(x_c_0)
    x_e_0 = np.array(x_e_0)
    areaBnds = np.array(areaBnds)

    # Evader position estimation
    x_e = x_e_0[:2] + timeStep * x_e_0[2:]

    # TPBVP solver time span
    tspan = np.linspace(0, timeStep, int(timeStep / 1E-1) + 1)

    # Initial guess
    guess = np.zeros((4, tspan.size))
    guess[0] = np.linspace(x_c_0[0], x_e[0], tspan.size)
    guess[1] = np.linspace(x_c_0[1], x_e[1], tspan.size)
    guess[2] = 0
    guess[3] = 0

    def eqns(t_mesh, y_mesh): 
        # Initialize return value variable. 
        dydt_mesh = np.zeros_like(y_mesh)

        # Equivalence threshold. 
        eps = 1e-3  

        # State space system. 
        A = np.zeros((2, 2))
        B = np.eye(2)

        # Mesh node loop. 
        for i in range(len(t_mesh)):
            # Isolate current mesh node state vector. 
            y = y_mesh[:, i]
            u = -np.sign(y[2:4]) * maxSpeed * (np.abs(y[2:4]) > eps)
            
            # Position and velocity constraints.
            if y[0] < areaBnds[0]:
                u[0] = maxSpeed
            elif y[0] > areaBnds[1]:
                u[0] = -maxSpeed
            if y[1] < areaBnds[2]:
                u[1] = maxSpeed
            elif y[1] > areaBnds[3]:
                u[1] = -maxSpeed

            dydt = np.hstack([A @ y[0:2] + B @ u, np.zeros(2)])
            
            dydt_mesh[:, i] = dydt

        # Return derivative mesh. 
        return dydt_mesh

    def boundaryConditions(ya, yb):
        """Boundary conditions."""
        return np.hstack([ya[0:2] - x_c_0,
                          yb[2:4] - (yb[0:2] - x_e)])

    # Solve BVP
    sol = solve_bvp(eqns, boundaryConditions, tspan, guess)

    # Extract solution
    y = sol.y
    return y[0:2, -1]



def mpc(x0, target, N, A, B, Q, R, E, W):
    """
    Model Predictive Control solver for pursuer trajectory optimization
    """
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

def ekf_func(z_k, x_kminus, P, Qk, Rk, acc_x_k, acc_y_k, dt_k):
    """
    Extended Kalman Filter function for state estimation
    """
    # Predict
    x_k_pred = x_kminus.copy()
    x_k_pred[0] = x_k_pred[0] + x_k_pred[2]*dt_k + 0.5*acc_x_k*dt_k**2
    x_k_pred[1] = x_k_pred[1] + x_k_pred[3]*dt_k + 0.5*acc_y_k*dt_k**2
    x_k_pred[2] = x_k_pred[2] + acc_x_k*dt_k
    x_k_pred[3] = x_k_pred[3] + acc_y_k*dt_k

    A_ekf = np.eye(4)
    A_ekf[0, 2] = dt_k
    A_ekf[1, 3] = dt_k
    
    P = A_ekf @ P @ A_ekf.T + Qk
    
    # Measurement update using GPS
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    y_res = z_k - H @ x_k_pred
    S = H @ P @ H.T + Rk
    K = P @ H.T @ np.linalg.inv(S)
    
    # Output
    x_kplus_hat = x_k_pred + K @ y_res
    P = (np.eye(4) - K @ H) @ P
    
    return x_kplus_hat, P

def compute_future_trajectories(x_evader_current, x_pursuer_current, u_evader, u_pursuer, N, A, B):
    """
    Compute future trajectories for the evader and pursuer given current states and inputs
    """
    x_future_evader = np.zeros((4, N+1))
    x_future_pursuer = np.zeros((4, N+1))
    
    x_future_evader[:, 0] = x_evader_current
    x_future_pursuer[:, 0] = x_pursuer_current
    
    for i in range(N):
        x_future_evader[:, i+1] = A @ x_future_evader[:, i] + B @ u_evader
        x_future_pursuer[:, i+1] = A @ x_future_pursuer[:, i] + B @ u_pursuer
    
    return x_future_evader, x_future_pursuer

def compute_optimal_targets(x_evader_current, x_pursuer_current, N, A, B, Ts):

    # For evader: compute optimal target that maximizes distance from pursuer
    # Compute a safe direction away from pursuer
    direction = x_evader_current[0:2] - x_pursuer_current[0:2]
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm  # compute cos, sin
    else:
        direction = np.array([1, 0])  # Default direction if same position
    
    # Calculate target position
    target_x = x_evader_current[0] + direction[0] * 20  # Look ahead
    target_y = x_evader_current[1] + direction[1] * 20
    
    # Keep current velocities for evader
    optimal_target_evader = np.array([target_x, target_y, x_evader_current[2], x_evader_current[3]])
    
    # For pursuer: compute optimal target that minimizes distance to evader
    # Predict evader's future position for interception
    predicted_evader_pos = x_evader_current[0:2] + x_evader_current[2:4] * N * Ts
    
    # Set pursuer's target to intercept the evader's predicted position
    optimal_target_pursuer = np.array([
        predicted_evader_pos[0], 
        predicted_evader_pos[1], 
        x_evader_current[2],  # Match evader's velocity for interception
        x_evader_current[3]
    ])

    return optimal_target_evader, optimal_target_pursuer
