import numpy as np
from utils import chaserTPBVP, mpc, ekf_func, compute_future_trajectories, compute_optimal_targets
import matplotlib.pyplot as plt


#################################
# Parameters
#################################
Ts = 0.1  # Time step
m = 1     # Mass
time_frames = 80
capture_radius = 2  # Distance at which pursuer catches evader
fence_width = 80    # x axis length
fence_height = 60   # y axis length
N = 7

# Init state
x_evader = np.zeros((4, time_frames))
x_pursuer = np.zeros((4, time_frames))
x_evader[:, 0] = np.array([-10, 10, 0, 0])  # Initial state [x, y, vx, vy]
x_pursuer[:, 0] = np.array([10, 10, 0, 0])
# E_p = np.vstack([np.eye(2*N), -np.eye(2*N)])
# W_p = np.ones(4*N) * 18
maxSpeed = 10
evader_max_speed = 8
u_evader = np.zeros(2)  # Initialize evader's acceleration
u_pursuer = np.zeros(2)  # Initialize pursuer's acceleration

A = np.array([
    [1, 0, Ts, 0],
    [0, 1, 0, Ts],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
B = np.zeros((4, 2))
B[2, 0] = Ts/m
B[3, 1] = Ts/m


###############################
# Game loop
###############################
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
    x_est_evader, P_evader = ekf_func(z_evader, x_evader[:, t], u_evader[0], u_evader[1], Ts)
    x_est_pursuer, P_pursuer = ekf_func(z_pursuer, x_pursuer[:, t], u_pursuer[0], u_pursuer[1], Ts)
    
    # Compute optimal targets for zero-sum game
    optimal_target_evader, optimal_target_pursuer = compute_optimal_targets(x_est_evader, x_est_pursuer, N, Ts)
    
    # Compute optimal control for evader and pursuer

    # Test parameters.

    areaBnds = np.array([-fence_width/2, fence_width/2, -fence_height/2, fence_height/2])

    u_evader, x_future_evader = mpc(x_est_evader, optimal_target_evader, Ts)
    retval = chaserTPBVP(x_est_pursuer[:2], x_est_evader, maxSpeed, areaBnds, Ts)
    # u_pursuer, x_future_pursuer = mpc(x_est_pursuer, optimal_target_pursuer, N, A, B, Q, R, E_p, W_p)

    
    # Update states
    x_evader[:, t+1] = A @ x_est_evader + B @ u_evader
    x_pursuer[:, t+1] = np.concatenate([retval, (retval - x_est_pursuer[:2])/Ts], axis=0)

    # Apply speed limit to evader
    next_evader_state = A @ x_est_evader + B @ u_evader
    evader_speed = np.sqrt(next_evader_state[2]**2 + next_evader_state[3]**2)
    if evader_speed > evader_max_speed:
        # Scale the velocity to the maximum speed
        scale_factor =  evader_max_speed / evader_speed
        next_evader_state[2] *= scale_factor
        next_evader_state[3] *= scale_factor
    x_evader[:, t+1] = next_evader_state
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


#############################
# Visualization
#############################
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


# Plot velocity
plt.figure(figsize=(8, 6))

plt.subplot(2, 2, 1)
plt.plot(np.arange(0, last_t+1), x_evader[2, :last_t+1], 'b-', label='Evader Vx')
plt.plot(np.arange(0, last_t+1), x_pursuer[2, :last_t+1], 'r-', label='Pursuer Vx')
plt.legend()
plt.grid(True)
plt.ylabel('X Velocity')

plt.subplot(2, 2, 3)
plt.plot(np.arange(0, last_t+1), x_evader[3, :last_t+1], 'b-', label='Evader Vy')
plt.plot(np.arange(0, last_t+1), x_pursuer[3, :last_t+1], 'r-', label='Pursuer Vy')
plt.legend()
plt.grid(True)
plt.xlabel('Time Step')
plt.ylabel('Y Velocity')



# plot pos
plt.subplot(2, 2, 2)
plt.plot(np.arange(0, last_t+1), x_evader[0, :last_t+1], 'b-', label='Evader X')
plt.plot(np.arange(0, last_t+1), x_pursuer[0, :last_t+1], 'r-', label='Pursuer X')  

plt.legend()
plt.grid(True)
plt.xlabel('Time Step')
plt.ylabel('X Position')

plt.subplot(2, 2, 4)
plt.plot(np.arange(0, last_t+1), x_evader[1, :last_t+1], 'b--', label='Evader Y')
plt.plot(np.arange(0, last_t+1), x_pursuer[1, :last_t+1], 'r--', label='Pursuer Y')
plt.legend()
plt.grid(True)
plt.xlabel('Time Step')
plt.ylabel('Y Position')
plt.savefig('position.jpg')

