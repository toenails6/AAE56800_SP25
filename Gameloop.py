import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_bvp
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
x_evader[:, 0] = np.array([-10, 0, 0, 1])  # Initial state [x, y, vx, vy]
x_pursuer[:, 0] = np.array([0, 0, -3, 0])

u_evader = np.zeros(2)  # Initialize evader's acceleration
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
E_e = np.vstack([np.eye(2*N), -np.eye(2*N)])
W_e = np.ones(4*N) * 10
E_p = np.vstack([np.eye(2*N), -np.eye(2*N)])
W_p = np.ones(4*N) * 18

# EKF parameters
P = np.eye(4)
Qk = 0.05 * np.eye(4)
Rk = 0.5 * np.eye(2)


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

            # Optimal control.
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

def compute_optimal_targets(x_evader_current, x_pursuer_current, N, A, B):

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

# Game loop
captured = False
capture_time = None
escaped = False
escape_time = None

for t in range(time_frames-1):
    # Current states
    evader_current = x_evader[:, t]
    pursuer_current = x_pursuer[:, t]
    
    # Measurements 
    z_evader = evader_current[0:2] + 0.2 * np.random.randn(2)
    z_pursuer = pursuer_current[0:2] + 0.2 * np.random.randn(2)
    
    # Estimate states using EKF
    x_est_evader, P_evader = ekf_func(z_evader, x_evader[:, t], P, Qk, Rk, u_evader[0], u_evader[1], Ts)
    x_est_pursuer, P_pursuer = ekf_func(z_pursuer, x_pursuer[:, t], P, Qk, Rk, u_pursuer[0], u_pursuer[1], Ts)
    
    # Compute optimal targets for zero-sum game
    optimal_target_evader, optimal_target_pursuer = compute_optimal_targets(x_est_evader, x_est_pursuer, N, A, B)
    
    # Compute optimal control for evader and pursuer

    # Test parameters.
    x_c_0 = np.array([-1, -1])
    x_e_0 = np.array([1, 0, 0, 1])
    maxSpeed = 10
    areaBnds = np.array([-fence_width/2, fence_width/2, -fence_height/2, fence_height/2])
    areaBnds[1] = -0.8

    u_evader, x_future_evader = mpc(x_est_evader, optimal_target_evader, N, A, B, Q, R, E_e, W_e)
    retval = chaserTPBVP(x_c_0, optimal_target_pursuer, maxSpeed, areaBnds, Ts)
    # u_pursuer, x_future_pursuer = mpc(x_est_pursuer, optimal_target_pursuer, N, A, B, Q, R, E_p, W_p)

    # Update states
    x_evader[:, t+1] = A @ x_est_evader + B @ u_evader
    x_pursuer[:, t+1] = A @ x_est_pursuer + B @ u_pursuer
    
    # Check for capture
    capture_distance = np.linalg.norm(x_evader[0:2, t+1] - x_pursuer[0:2, t+1])
    if capture_distance < capture_radius:
        print(f'Evader captured at time step {t+1}!')
        captured = True
        capture_time = t+1
        break
    
    # Check for evader out of fence
    if not (-fence_width/2 <= x_evader[0, t+1] <= fence_width/2 and -fence_height/2 <= x_evader[1, t+1] <= fence_height/2):
        print(f'Evader escaped at time step {t+1}!')
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
plt.plot(x_evader[0, :last_t+1], x_evader[1, :last_t+1], 'b-', linewidth=1, label='Evader Path')
plt.plot(x_pursuer[0, :last_t+1], x_pursuer[1, :last_t+1], 'r-', linewidth=1, label='Pursuer Path')
plt.plot(x_evader[0, 0], x_evader[1, 0], 'bo', markersize=8, markerfacecolor='b', label='Evader Start')
plt.plot(x_pursuer[0, 0], x_pursuer[1, 0], 'ro', markersize=8, markerfacecolor='r', label='Pursuer Start')
plt.plot(x_evader[0, last_t], x_evader[1, last_t], 'bx', markersize=8, linewidth=2, label='Evader End')
plt.plot(x_pursuer[0, last_t], x_pursuer[1, last_t], 'rx', markersize=8, linewidth=2, label='Pursuer End')

# Plot fence
fence_x = [-fence_width/2, fence_width/2, fence_width/2, -fence_width/2, -fence_width/2]
fence_y = [-fence_height/2, -fence_height/2, fence_height/2, fence_height/2, -fence_height/2]
plt.plot(fence_x, fence_y, 'g--', linewidth=2, label='Fence')

# Draw capture radius if it happened
if captured:
    circle = plt.Circle((x_pursuer[0, last_t], x_pursuer[1, last_t]), capture_radius, color='r', fill=False, linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    plt.text(x_pursuer[0, last_t], x_pursuer[1, last_t]+3, f'Captured at t={last_t}', horizontalalignment='center')

# Mark escape point if it happened
if escaped:
    plt.text(x_evader[0, last_t], x_evader[1, last_t]+3, f'Escaped at t={last_t}', horizontalalignment='center')

plt.title('Pursuer-Evader Game with Zero-Sum Game Approach')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('pursuer_evader_game.png')
plt.show()

plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.plot(np.arange(0, last_t+1), x_evader[2, :last_t+1], 'b-', label='Evader Vx')
plt.plot(np.arange(0, last_t+1), x_pursuer[2, :last_t+1], 'r-', label='Pursuer Vx')
plt.legend()
plt.grid(True)
plt.ylabel('X Velocity')

plt.subplot(2, 1, 2)
plt.plot(np.arange(0, last_t+1), x_evader[3, :last_t+1], 'b-', label='Evader Vy')
plt.plot(np.arange(0, last_t+1), x_pursuer[3, :last_t+1], 'r-', label='Pursuer Vy')
plt.legend()
plt.grid(True)
plt.xlabel('Time Step')
plt.ylabel('Y Velocity')

plt.savefig('velocity.jpg')
