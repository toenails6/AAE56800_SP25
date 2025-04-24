# AAE56800_SP25
AAE56800 SP25 UAV Chaser-Evader Game: Control, Estimation, and Learning under Wind Disturbances.

# Chaser-Evader game theory. 
Two drones utilize two different optimal control algorithms: TPBVP and MPC. 
One attempting to evade being captured, while the other attempting to capture the evading drone. 
The entire simulation is separated into numerous time steps, at the start of which each algorithm updates its target's states. 
In the duration of one time step, both algorithms would predict their targets' flight within this time step and find the optimal control to achieve the goal of evasion or capture. 
At the beginning of the next time step, the actual states of the target drones would be given to both algorithms, and the game begins anew. 
The process continues until time ends or one drone achieves its goal and is declared winner. 

In example, the chaser is given the evader's initial states at the start of a new time step. 
For the rest of the duration of this time step, the chaser is not informed of the evader's real-time actual states, and will rely on its own algorithmic predictions of the evader's maneuvers to find the optimal course for capture. 
The evader's actual states will be revealed to the chaser at the beginning of the next time step, and the cycle continues until either the chaser runs out of time, or the evader is forced within a defined vicinity to the chaser. 

# TPBVP Problem formulation. 
Chaser-Evader state space: 
```math
\dot{x} = Ax+Bu
```
```math
A = \begin{bmatrix}
    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
```
```math
x = \begin{bmatrix}
    x_c & y_c & x_e & y_e & v_{x,c} & v_{y,c} & v_{x,e} & v_{y,e}
\end{bmatrix}^T
```
```math
B = \begin{bmatrix}
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 \\
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1
\end{bmatrix}
```
```math
u = \begin{bmatrix}
    u_{x,c} & u_{y,c} & u_{x,e} & u_{y,e}
\end{bmatrix}^T
```
Hamiltonian:
```math
H=
```
