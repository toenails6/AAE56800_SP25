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
    0 & I_4 \\
    0 & 0 
\end{bmatrix}
```
```math
x = \begin{bmatrix}
    x_c & y_c & x_e & y_e & v_{x,c} & v_{y,c} & v_{x,e} & v_{y,e}
\end{bmatrix}^T
```
```math
B = \begin{bmatrix}
    0 \\ I_4
\end{bmatrix}
```
```math
u = \begin{bmatrix}
    u_{x,c} & u_{y,c} & u_{x,e} & u_{y,e}
\end{bmatrix}^T
```

Zero-Sum Differential Optimal Control Problem Formulation: 
```math
\begin{aligned}
    \arg \quad & 
    J = \phi(x(t_f)) + 
    \int_{t_0}^{t_f}L(x(t_f), u(t))\cdot dt \\ 
    \textrm{where} \quad & 
    \phi(x(t_f)) = \frac{1}{2}[(x_c(t_f)-x_e(t_f))^2+(y_c(t_f)-y_e(t_f))^2] \\ &
    L(u(t)) = \frac{1}{2}(u_{x,c}^2+u_{y,c}^2+u_{x,e}^2+u_{y,e}^2) \\
    \textrm{s.t.} \quad & 
    \dot{x} = Ax+Bu \\ & 
    x_l \leq x_c \leq x_u \\ &
    y_l \leq y_c \leq y_u \\ &
    x_l \leq x_e \leq x_u \\ &
    y_l \leq y_e \leq y_u \\ &
    y_{x,c}^2 + y_{y,c}^2 \leq V^2 \\ & 
    y_{x,e}^2 + y_{y,e}^2 \leq V^2 \\ &
    u_l \leq u \leq u_u
\end{aligned}
```

The Hamiltonian is then:
```math
H = L(u(t)) + \lambda^T(Ax+Bu) + \mu^TC + \nu^TS + \alpha^TU
```
where: 
```math
C = \begin{bmatrix}
    x_l - x_c(t) \\ 
    x_c(t) - x_u \\ 
    y_l - y_c(t) \\ 
    y_c(t) - y_u \\ 
    x_l - x_e(t) \\ 
    x_e(t) - x_u \\ 
    y_l - y_e(t) \\ 
    y_e(t) - y_u
\end{bmatrix}
```
```math
S = \begin{bmatrix}
    y_{x,c}^2 + u_{y,c}^2 - V^2 \\ 
    y_{x,e}^2 + u_{y,e}^2 - V^2
\end{bmatrix}
```
```math
U\begin{bmatrix}
    u_l - u_{x,c}(t) \\ 
    u_{x,c}(t) - u_u \\ 
    u_l - u_{y,c}(t) \\ 
    u_{y,c}(t) - u_u \\ 
    u_l - u_{x,e}(t) \\ 
    u_{x,e}(t) - u_u \\ 
    u_l - u_{y,e}(t) \\ 
    u_{y,e}(t) - u_u
\end{bmatrix}
```

We thus have the State dynamics: 
```math
\dot{x} = \frac{\partial H}{\partial \lambda} = Ax+Bu
```

The Co-state dynamics: 
```math
\dot{\lambda}=-\frac{\partial H}{\partial x} = \begin{bmatrix}
    \mu_1 - \mu_2 \\ 
    \mu_3 - \mu_4 \\ 
    \mu_5 - \mu_6 \\ 
    \mu_7 - \mu_8 \\ 
    -\lambda_1 \\ 
    -\lambda_2 \\ 
    -\lambda_3 \\ 
    -\lambda_4
\end{bmatrix}
```

Control Optimality: 
```math
\frac{\partial H}{\partial u} = \begin{bmatrix}
    u_{x,c} + \lambda_5 + 2\nu_1 u_{x,c} \\ 
    u_{y,c} + \lambda_6 + 2\nu_1 u_{y,c} \\ 
    -u_{x,e} + \lambda_7 + 2\nu_2 u_{x,e} \\ 
    -u_{y,e} + \lambda_8 + 2\nu_2 u_{y,e}
\end{bmatrix} = 0
```
