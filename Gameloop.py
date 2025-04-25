import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular
from qpsolvers import solve_qp

# Parameters
Ts = 0.1  # Time step
m = 1     # Mass
time_frames = 60
capture_radius = 2  # Distance at which pursuer catches evader
fence_width = 80    # x axis length
fence_height = 60   # y axis length
N = 7

# Initialize state
x_evader = np.zeros((4, time_frames))
x_pursuer = np.zeros((4, time_frames))
x_evader[:, 0] = np.array([-10, 20, 3, 0])  # Initial state [x, y, vx, vy]
x_pursuer[:, 0] = np.array([30, 20, -3, 0])

u_evader = np.array([0, -5])  # Fixed acceleration for evader
u_pursuer = np.zeros(2)  # Initialize pursuer's acceleration


# System dynamics matrices
A = np.array([
    [1, 0, Ts, 0],
    [0, 1, 0, Ts],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

B = np.zeros((4, 2))
B[2, 0] = Ts/m
B[3, 1] = Ts/m

# MPC Parameters
Q = np.diag([20, 20, 1, 1])
R = np.diag([1, 1])

# Constraint matrices
E = np.vstack([np.eye(2*N), -np.eye(2*N)])
W = np.ones(4*N) * 10

# EKF parameters
P = np.eye(4)
Qk = 0.05 * np.eye(4)
Rk = 0.5 * np.eye(2)

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

# Game loop
captured = False
capture_time = None
capture_possible = False
escaped = False
escape_time = None

for t in range(time_frames-1):
    # Current states
    evader_current = x_evader[:, t]
    pursuer_current = x_pursuer[:, t]
    
    # Measurements 
    z_evader = evader_current[0:2] + 0.2 * np.random.randn(2)
    z_pursuer = pursuer_current[0:2] + 0.2 * np.random.randn(2)
    x_est_evader = ekf_func(z_evader, x_evader[:, t], P, Qk, Rk, u_evader[0], u_evader[1], Ts)[0]
    x_est_pursuer = ekf_func(z_pursuer, x_pursuer[:, t], P, Qk, Rk, u_pursuer[0], u_pursuer[1], Ts)[0]
    x_pursuer[:, t+1] = A @ x_est_pursuer + B @ u_pursuer
    x_evader[:, t+1] = A @ x_est_evader + B @ u_evader

    # Evader's future trajectory
    x_future_evader = np.zeros((4, N))
    x_future_evader[:, 0] = x_evader[:, t+1]
    x_future_pursuer = np.zeros((4, N))
    x_future_pursuer[:, 0] = x_pursuer[:, t+1]

    for i in range(1, N):
        x_future_evader[:, i] = A @ x_future_evader[:, i-1] + B @ u_evader
        x_future_pursuer[:, i] = A @ x_future_pursuer[:, i-1] + B @ u_pursuer

    # Optimal interception point for pursuer
    optimal_target_pursuer = x_est_evader

    scan_radius_pursuer = np.linalg.norm(x_future_pursuer[0:2, -1] - x_est_pursuer[0:2])
    capture_distance_future = np.linalg.norm(x_future_evader[0:2, -1] - x_est_pursuer[0:2])
    if capture_distance_future <= scan_radius_pursuer:  # If pursuer can reach evader position
        optimal_target_pursuer = x_future_evader[:, -1]  
        capture_possible = True
        print("Capture possible at step", t)

    # Compute optimal control for pursuer
    u_pursuer, x_pursuer = mpc(x_est_pursuer, optimal_target_pursuer, N, A, B, Q, R, E, W)

    # Check for capture
    capture_distance = np.linalg.norm(x_evader[0:2, t] - x_pursuer[0:2, t])
    if capture_distance < capture_radius:
        print(f'Evader captured at time step {t}!')
        captured = True
        capture_time = t
        break

    # check for evader out of fence
    if not (-fence_width/2 <= x_evader[0, t] <= fence_width/2 and -fence_height/2 <= x_evader[1, t] <= fence_height/2):
        print(f'Evader escaped at time step {t}!')
        escaped = True
        escape_time = t+1
        break

# Visualization
if captured or escaped:
    if escaped:
        last_t = escape_time
    else:
        last_t = capture_time
else:
    last_t = time_frames-1

plt.figure(figsize=(8, 6))
plt.plot(x_evader[0, :last_t+1], x_evader[1, :last_t+1], 'bp-', linewidth=1, label='Evader Path')
plt.plot(x_pursuer[0, :last_t+1], x_pursuer[1, :last_t+1], 'rp-', linewidth=1, label='Pursuer Path')
plt.plot(x_evader[0, 0], x_evader[1, 0], 'bo', markersize=8, markerfacecolor='b', label='Evader Start')
plt.plot(x_pursuer[0, 0], x_pursuer[1, 0], 'ro', markersize=8, markerfacecolor='r', label='Pursuer Start')
plt.plot(x_evader[0, last_t], x_evader[1, last_t], 'bx', markersize=8, linewidth=2, label='Evader End')
plt.plot(x_pursuer[0, last_t], x_pursuer[1, last_t], 'rx', markersize=8, linewidth=2, label='Pursuer End')

# Plot fence
fence_x = [-fence_width/2, fence_width/2, fence_width/2, -fence_width/2, -fence_width/2]
fence_y = [-fence_height/2, -fence_height/2, fence_height/2, fence_height/2, -fence_height/2]
plt.plot(fence_x, fence_y, 'g--', linewidth=2, label='Fence')


# Draw capture if it happened
if captured or escaped:
    plt.plot(x_evader[0, last_t], x_evader[1, last_t], 'bx', markersize=8)

plt.title('Pursuer-Evader Game')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
plt.savefig('pursuer_evader_game.jpg')