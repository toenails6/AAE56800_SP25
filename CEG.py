import numpy as np
import math
import tf
from tf.transformations import quaternion_from_euler
import rospy
from geometry_msgs.msg import PoseStamped

class home_pos():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.theta = 0
        self.pitch = 0
        self.row = 0
        self.mg_heading = 0

def generate_precise_figure8_waypoints(radius, altitude, num_points):
    """
    Generates a continuous figure-8 with:
    - Left loop: CCW from (0,0) around center (-radius,0)
    - Right loop: CW from (0,0) around center (+radius,0)
    Both start and end at (0,0)
    """
    waypoints = []

    # Left loop (CCW) around center (-r, 0)
    cx_left, cy_left = -radius, 0
    for theta in np.linspace(0, 2 * np.pi, num_points, endpoint=False):
        x = cx_left + radius * np.cos(theta)
        y = cy_left + radius * np.sin(theta)
        waypoints.append((x, y, altitude))

    waypoints.append((0.0, 0.0, altitude))  # return to center

    # Right loop (CW) around center (+r, 0)
    cx_right, cy_right = radius, 0
    for theta in np.linspace(0, 2 * np.pi, num_points, endpoint=False):
        x = cx_right + radius * np.cos(-theta - np.pi)
        y = cy_right + radius * np.sin(-theta - np.pi)
        waypoints.append((x, y, altitude))

    waypoints.append((0.0, 0.0, altitude))  # final return to center

    return waypoints


# EKF function
Q = np.diag([1e-2]*3 + [1e-1]*3 + [1e-2]*3)
R = np.diag(np.array([0.5, 0.5, 0.5, 5 * np.pi / 180, 5 * np.pi / 180, 5 * np.pi / 180])**2)
Pplus0 = 1e-2 * np.eye(9)
Pplus1 = 1e-2 * np.eye(9)
def packed_EKF(Xplus, Pplus, r, Q, y, acc_body, dt):
    def ekf_dynamics(x, acc_body, dt):
        yaw = x[6, 0] if x.ndim == 2 else x[6]
        pitch= x[7, 0] if x.ndim == 2 else x[7]
        roll = x[8, 0] if x.ndim == 2 else x[8]
        
        RM = np.array([
            [np.cos(pitch) * np.cos(yaw),
            np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw),
            np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw)],
            [np.cos(pitch) * np.sin(yaw),
            np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw),
            np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw)],
            [-np.sin(pitch),
            np.sin(roll) * np.cos(pitch),
            np.cos(roll) * np.cos(pitch)]
        ])
        acc_world = RM @ acc_body
        acc_world[2] = 0

        pos = x[0:3] + x[3:6] * dt + 0.5 * acc_world * dt**2
        vel = x[3:6] + acc_world * dt
        yaw_next = x[6:9]

        return np.concatenate((pos, vel, yaw_next))

    def ekf_jacobian_A(dt):
        A = np.eye(9)
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt
        return A

    def ekf_measurement(x):
        return np.array([x[0], x[1], x[2], x[6], x[7], x[8]])

    def ekf_jacobian_H():
        H = np.zeros((6, 9))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 6] = 1
        H[4, 7] = 1
        H[5, 8] = 1
        return H

    # Prediction step
    Xminus = ekf_dynamics(Xplus.flatten(), acc_body, dt)
    A = ekf_jacobian_A(dt)
    Pminus = A @ Pplus @ A.T + Q

    # Update step
    hx = ekf_measurement(Xminus)
    if hx.ndim == 1:
        hx = hx.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if Xminus.ndim == 1:
        Xminus = Xminus.reshape(-1, 1)
    H = ekf_jacobian_H()
    S = H @ Pminus @ H.T + r
    K = Pminus @ H.T @ np.linalg.inv(S)
    Xplus = Xminus + K @ (y - hx)
    Pplus = (np.eye(9) - K @ H) @ Pminus

    return Xplus, Pplus


if __name__ == "__main__":
    import time
    from FlightControlNode import FlightControlNode
    import utm

    # === Generate Example Trajectory for UAV0 ===
    UAV0_wp = generate_precise_figure8_waypoints(5,3,100)
    UAV1_wp = generate_precise_figure8_waypoints(5,3,100)

    # === setup UAV control node ===
    FCN0 = FlightControlNode("/uav0/")
    FCN1 = FlightControlNode("/uav1/")

    # === get UAV ground truth data ===
    FCN0.model_name = 'iris0'
    FCN1.model_name = 'iris1'
    
    # === sleep 1 second establishing mavros service ===
    time.sleep(1)

    # === record UAV GPS home position ===
    UAV0_HomePos = home_pos()
    UAV0_HomePos.x = FCN0.curr_x
    UAV0_HomePos.y = FCN0.curr_y
    UAV0_HomePos.z = FCN0.curr_z
    UAV0_HomePos.mg_heading = FCN0.mg_heading

    UAV1_HomePos = home_pos()
    UAV1_HomePos.x = FCN1.curr_x
    UAV1_HomePos.y = FCN1.curr_y
    UAV1_HomePos.z = FCN1.curr_z
    UAV1_HomePos.mg_heading = FCN1.mg_heading

    # === set max velocity [NEEDS to be FIX ignore for now] ===
    ## FCN0.set_max_velocity(1)
    ## FCN1.set_max_velocity(1)

    # === Take off all UAVs ===
    UAV0_takoff_alt = 3
    FCN0.auto_takeoff(UAV0_takoff_alt)
    print("UAV0 Take Off")

    UAV1_takoff_alt = 3
    FCN1.auto_takeoff(UAV1_takoff_alt)
    print("UAV1 Take Off")

    # === Wait 5 seconds for all UAVs to reach disired altitude ===
    time.sleep(5)

    UAV0_prev_loc_x = 0
    UAV0_prev_loc_y = 0
    UAV0_prev_loc_z = UAV0_takoff_alt # relective local z, UAV0 compare to origin
    UAV0_prev_theta = UAV0_HomePos.mg_heading  # magnetic heading

    UAV1_prev_loc_x = UAV1_HomePos.x-UAV0_HomePos.x # relective local x, UAV1 compare to UAV0
    UAV1_prev_loc_y = UAV1_HomePos.y-UAV0_HomePos.y # relective local y, UAV1 compare to UAV0
    UAV1_prev_loc_z = UAV1_takoff_alt # relective local z, UAV1 compare to origin
    UAV1_prev_theta = UAV1_HomePos.mg_heading  # magnetic heading

    UAV0_Xplus = np.array([[UAV0_prev_loc_x], [UAV0_prev_loc_y], [3], [0], [0], [0], [0], [0], [0]])
    UAV1_Xplus = np.array([[UAV1_prev_loc_x], [UAV1_prev_loc_y], [3], [0], [0], [0], [0], [0], [0]])
    UAV0_EKF_history = []
    UAV0_GPS_history = []
    UAV0_GTT_history = []

    # === Begin Casher Evader Game ===
    print('Begin Casher Evader Game')
    for i, (wp0, wp1) in enumerate(zip(UAV0_wp, UAV1_wp)):
        # refresh rate to send new waypoint 0.1s
        dt = 0.2
        time.sleep(dt)

        # === get current measurment from sensor ===
        UAV0_Curr_GPS_x = FCN0.lat  # GPS raw latitude (with noise)
        UAV0_Curr_GPS_y = FCN0.lon  # GPS raw longitude (with noise)
        UAV0_Curr_GPS_z = FCN0.alt  # GPS raw alatitude (with noise)
        UAV0_IMU_body_ax = FCN0.ax # IMU body frame accel. x (with noise)
        UAV0_IMU_body_ay = FCN0.ay # IMU body frame accel. y (with noise)
        UAV0_IMU_body_az = FCN0.az # IMU body frame accel. z (with noise)
        UAV0_world_yaw = FCN0.yaw  # PX4 EKF processed yaw
        UAV0_world_pitch = FCN0.pitch  # PX4 EKF processed pitch
        UAV0_world_roll = FCN0.roll  # PX4 EKF processed roll
        UAV0_Curr_theta = FCN0.mg_heading  # magnetic heading (with noise)

        # print('ax,ay,az', UAV0_IMU_body_ax,UAV0_IMU_body_ay,UAV0_IMU_body_az)
        # print('roll, pitch, yaw', UAV0_world_roll,UAV0_world_pitch,UAV0_world_yaw)
        # print('theta', UAV0_Curr_theta)
        # print('GTT0: ',FCN0.position)
        UAV0_GTT_history.append(np.array([FCN0.position.x, FCN0.position.y]))

        UAV1_Curr_GPS_x = FCN1.lat  # GPS raw latitude (with noise)
        UAV1_Curr_GPS_y = FCN1.lon  # GPS raw longitude (with noise)
        UAV1_Curr_GPS_z = FCN1.alt  # GPS raw alatitude (with noise)
        UAV1_IMU_body_ax = FCN1.ax # IMU body frame accel. x (with noise)
        UAV1_IMU_body_ay = FCN1.ay # IMU body frame accel. y (with noise)
        UAV1_IMU_body_az = FCN1.az # IMU body frame accel. z (with noise)
        UAV1_world_yaw = FCN1.yaw  # PX4 EKF processed yaw
        UAV1_world_pitch = FCN1.pitch  # PX4 EKF processed pitch
        UAV1_world_roll = FCN1.roll  # PX4 EKF processed roll
        UAV1_Curr_theta = FCN1.mg_heading  # magnetic heading (with noise)
        
        # === convert GPS [lat,lon] to local frame [0,0] raw position ===
        UAV0_utm_x, UAV0_utm_y, UAV0_zone, UAV0_ut = utm.from_latlon(UAV0_Curr_GPS_x,UAV0_Curr_GPS_y)
        UAV0_raw_loc_x = UAV0_utm_x - UAV0_HomePos.x # UAV0 local x with noise
        UAV0_raw_loc_y = UAV0_utm_y - UAV0_HomePos.y # UAV0 local y with noise
        print('GPS Raw Loc',UAV0_raw_loc_x,UAV0_raw_loc_y)
        UAV0_GPS_history.append(np.array([UAV0_raw_loc_x, UAV0_raw_loc_y]))

        UAV1_utm_x, UAV1_utm_y, UAV1_zone, UAV1_ut = utm.from_latlon(UAV1_Curr_GPS_x,UAV1_Curr_GPS_y)
        UAV1_raw_loc_x = UAV1_utm_x - UAV1_HomePos.x # UAV1 local x with noise
        UAV1_raw_loc_y = UAV1_utm_y - UAV1_HomePos.y # UAV1 local y with noise

        
        # +++ Start of EKF algorithm +++
        UAV0_y = np.array([UAV0_raw_loc_x, UAV0_raw_loc_y, 3, UAV0_world_yaw, UAV0_world_pitch, UAV0_world_roll]).reshape(-1, 1)
        UAV0_acc_body = np.array([UAV0_IMU_body_ax, UAV0_IMU_body_ay, UAV0_IMU_body_az])
        UAV0_Xplus, Pplus0 = packed_EKF(UAV0_Xplus, Pplus0, R, Q, UAV0_y, UAV0_acc_body, dt)

        UAV1_y = np.array([UAV1_raw_loc_x, UAV1_raw_loc_y, 3, UAV1_world_yaw, UAV1_world_pitch, UAV1_world_roll]).reshape(-1, 1)
        UAV1_acc_body = np.array([UAV1_IMU_body_ax, UAV1_IMU_body_ay, UAV1_IMU_body_az])
        UAV1_Xplus, Pplus1 = packed_EKF(UAV1_Xplus, Pplus1, R, Q, UAV1_y, UAV1_acc_body, dt)

        UAV0_EKF_history.append(UAV0_Xplus.copy())
        # --- END of EKF algorithm ---
        
        # === Load wapoints for both UAV ===
        # EKF output usage: 
        # UAV0_Xplus = [x; y; z; vx; vy; vz; yaw; pitch; roll]
        # UAV1_Xplus = [x; y; z; vx; vy; vz; yaw; pitch; roll] 
        # Let MPC control UAV0, and TPBVP control UAV1
        # Initial position UAV0: [0; 0; 3] (m; local coordinate/world frame)
        # Initial position UAV1: [0; 5; 3] (m; local coordinate/world frame) could change their initial distance larger later
        
        # +++ Insert MPC algorithm here +++
        (x0 ,y0, z0) = wp0 # replace wp0 with your algorithm, could use UAV0_Xplus and UAV1_Xplus as input

        # +++ Insert TPBVP algorithm here +++
        (x1 ,y1, z1) = wp1 # replace wp1 with your algorithm, could use UAV0_Xplus and UAV1_Xplus as input

        # === Compute theta to face direction of motion ===
        UAV0_dx = x0 - UAV0_prev_loc_x
        UAV0_dy = y0 - UAV0_prev_loc_y
        UAV0_theta = math.atan2(UAV0_dy, UAV0_dx)

        UAV1_dx = x1 - UAV1_prev_loc_x
        UAV1_dy = y1 - UAV1_prev_loc_y
        UAV1_theta = math.atan2(UAV1_dy, UAV1_dx)

        # === Save current local position for next time ===
        UAV0_prev_loc_x = UAV0_raw_loc_x
        UAV0_prev_loc_y = UAV0_raw_loc_y

        UAV1_prev_loc_x = UAV1_raw_loc_x
        UAV1_prev_loc_y = UAV1_raw_loc_y

        # === Convert theta to quaternion ===
        q0 = tf.transformations.quaternion_from_euler(0, 0, UAV0_theta)
        q1 = tf.transformations.quaternion_from_euler(0, 0, UAV1_theta)

        print("Proceed to Next Waypoint ",i)

        print('EKF_x,y,z: ',UAV0_Xplus[0],UAV0_Xplus[1],UAV0_Xplus[2])


        FCN0.local_offboard_ctrl(x0,y0,z0,q0)
        FCN1.local_offboard_ctrl(x1,y1,z1,q1)

    time.sleep(3)
    print('iris0 position',FCN0.position)
    print('iris1 position',FCN1.position)

import matplotlib.pyplot as plt

# Convert lists to arrays
EKF = np.array(UAV0_EKF_history).squeeze()       # shape (N, 9)
GPS = np.array(UAV0_GPS_history)                 # shape (N, 2)
GTT = np.array(UAV0_GTT_history)                 # shape (N, 3)

# Extract x and y
EKF_x, EKF_y = EKF[:, 0], EKF[:, 1]
GPS_x, GPS_y = GPS[:, 0], GPS[:, 1]
GTT_x, GTT_y = GTT[:, 0], GTT[:, 1]

# Plot
plt.figure(figsize=(8,6))
plt.plot(GTT_x, GTT_y, 'k--', label='Ground Truth (GTT)')
plt.plot(GPS_x, GPS_y, 'b:', label='GPS Raw')
plt.plot(EKF_x, EKF_y, 'r-', label='EKF Estimate')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('UAV0 Trajectory Comparison')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

    
