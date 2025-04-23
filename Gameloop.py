import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular

# Parameters
Ts = 0.1  # Time step
m = 1     # Mass
time_frames = 60
capture_radius = 3  # Distance at which pursuer catches evader

# Initialize state
x_evader = np.zeros((4, time_frames))
x_pursuer = np.zeros((4, time_frames))
x_evader[:, 0] = np.array([10, 20, 3, 0])  # Initial state [x, y, vx, vy]
x_pursuer[:, 0] = np.array([50, 50, -3, -3])

a_evader = np.array([0, -5])  # Fixed acceleration for evader

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
Q = np.diag([10, 10, 1, 1])
R = np.diag([1, 1])
N = 10

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
    u = solve_qp(L, F @ (x0 - target), E, W)
    
    # Prediction steps
    uMPC = u[:2]  # First control input
    xMPC = np.zeros((4, N+1))
    xMPC[:, 0] = x0
    
    for i in range(N):
        xMPC[:, i+1] = A @ xMPC[:, i] + B @ u[i*2:(i+1)*2]
    
    return uMPC, xMPC

def solve_qp(H, f, A_ineq=None, b_ineq=None):
    """
    Simplified Quadratic Programming solver - this is a placeholder
    A full implementation would use an active set or interior point method
    
    This simplified version just uses scipy's minimize function
    """
    from scipy.optimize import minimize
    
    n = f.shape[0]
    
    def objective(x):
        return 0.5 * x.T @ H @ x + f.T @ x
    
    # Define constraints
    constraints = []
    if A_ineq is not None and b_ineq is not None:
        for i in range(A_ineq.shape[0]):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: b_ineq[i] - A_ineq[i, :] @ x
            })
    
    # Initial guess
    x0 = np.zeros(n)
    
    result = minimize(objective, x0, constraints=constraints, method='SLSQP')
    return result.x

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

for t in range(time_frames-1):
    # Current states
    evader_current = x_evader[:, t]
    pursuer_current = x_pursuer[:, t]
    
    # Measurements 
    z_evader = evader_current[0:2] + 0.2 * np.random.randn(2)
    z_pursuer = pursuer_current[0:2] + 0.2 * np.random.randn(2)
    
    # Update evader state using fixed acceleration
    u_evader = a_evader
    x_evader[:, t+1] = A @ evader_current + B @ u_evader
    
    # Estimate evader's future trajectory for pursuer planning
    evader_pred = x_evader[:, t+1]
    evader_future = np.zeros((4, N))
    evader_future[:, 0] = evader_pred
    
    # Predict evader's future positions based on constant acceleration
    for i in range(1, N):
        evader_future[:, i] = A @ evader_future[:, i-1] + B @ u_evader
    
    # Determine optimal interception point for pursuer
    optimal_target = evader_future[:, N-1]
    
    # Check if capture is possible within prediction horizon
    capture_possible = False
    for step in range(N):
        distance = np.linalg.norm(evader_future[0:2, step] - pursuer_current[0:2])
        time_to_reach = (step + 1) * Ts
        if distance / time_to_reach < 10:  # If pursuer can reach evader position
            optimal_target = evader_future[:, step]
            capture_possible = True
            break
    
    # If capture doesn't seem possible within horizon, aim for final predicted position
    if not capture_possible:
        optimal_target = evader_future[:, N-1]
    
    # Use MPC to compute optimal control for pursuer
    u_pursuer, x_pred = mpc(pursuer_current, optimal_target, N, A, B, Q, R, E, W)
    
    # Update pursuer state
    x_pursuer[:, t+1] = A @ pursuer_current + B @ u_pursuer
    
    # Check for capture
    distance = np.linalg.norm(x_evader[0:2, t+1] - x_pursuer[0:2, t+1])
    if distance < capture_radius:
        print(f'Evader captured at time step {t+1}!')
        captured = True
        capture_time = t+1
        break

# Visualization
last_t = capture_time if captured else time_frames-1

plt.figure(figsize=(10, 8))
plt.plot(x_evader[0, :last_t+1], x_evader[1, :last_t+1], 'b-', linewidth=2, label='Evader Path')
plt.plot(x_pursuer[0, :last_t+1], x_pursuer[1, :last_t+1], 'r-', linewidth=2, label='Pursuer Path')
plt.plot(x_evader[0, 0], x_evader[1, 0], 'bo', markersize=10, markerfacecolor='b', label='Evader Start')
plt.plot(x_pursuer[0, 0], x_pursuer[1, 0], 'ro', markersize=10, markerfacecolor='r', label='Pursuer Start')
plt.plot(x_evader[0, last_t], x_evader[1, last_t], 'bx', markersize=10, linewidth=2, label='Evader End')
plt.plot(x_pursuer[0, last_t], x_pursuer[1, last_t], 'rx', markersize=10, linewidth=2, label='Pursuer End')

# Draw capture if it happened
if captured:
    print('Captured')
    plt.plot(x_evader[0, last_t], x_evader[1, last_t], 'ko', markersize=15)

plt.title('Pursuer-Evader Game')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
plt.savefig('pursuer_evader_game.jpg')
