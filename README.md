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

# Problem formulation
Chaser state space model: 
$$
\dot{x} = Ax+Bu
$$
$$
A = \begin{bmatrix}
    0 & 0 \\
    0 & 0 
\end{bmatrix} \quad
B = \begin{bmatrix}
    1 & 0 \\ 0 & 1
\end{bmatrix}
$$
$$
x = \begin{bmatrix}
    p_{x,c} & p_{y,c}
\end{bmatrix}^T \quad
u = \begin{bmatrix}
    u_{x,c} & u_{y,c}
\end{bmatrix}^T
$$

Evader state space model: 
$$
\dot{x} = Ax+Bu
$$
$$
A = \begin{bmatrix}
    0 & I_2 \\
    0 & 0 
\end{bmatrix} \quad
B = \begin{bmatrix}
    0 \\ I_2
\end{bmatrix}
$$
$$
x = \begin{bmatrix}
    p_{x,e} & p_{y,e} & v_{x,e} & v_{y,e}
\end{bmatrix}^T \quad
u = \begin{bmatrix}
    u_{x,e} & u_{y,e}
\end{bmatrix}^T
$$
Let $T_s$ denote the length of a time step. 

Chaser optimal control problem formulation: 
$$
\begin{aligned}
    \min_{u} \quad & 
    J = \phi(x(t_f)) \\ 
    \textrm{where} \quad & 
    \phi(x(t_f)) = \frac{1}{2}[(p_{x,c}(t_f)-p_{x,e}(t_f))^2+(p_{y,c}(t_f)-p_{y,e}(t_f))^2] \\ &
    p_e(t_f) = p_e(0) + v_e\cdot T_s \\
    \textrm{s.t.} \quad & 
    \dot{x} = Ax+Bu \\ & 
    x_l \leq p_x \leq x_u \\ &
    y_l \leq p_y \leq y_u \\ &
    -V \leq u \leq V
\end{aligned}
$$

Evader optimal control problem formulation: 
$$
\begin{aligned}
    \max_{u} \quad & 
    J = \phi(x(t_f)) \\ 
    \textrm{where} \quad & 
    \phi(x(t_f)) = \frac{1}{2}[(p_{x,c}(t_f)-p_{x,e}(t_f))^2+(p_{y,c}(t_f)-p_{y,e}(t_f))^2] \\ &
    p_c(t_f) = p_c(0) + v_c\cdot T_s \\
    \textrm{s.t.} \quad & 
    \dot{x} = Ax+Bu \\ & 
    x_l \leq p_x \leq x_u \\ &
    y_l \leq p_y \leq y_u \\ &
    -V \leq v_{x,e} \leq V \\ &
    -V \leq v_{y,e} \leq V \\ &
    -G \leq u_{x,e} \leq G \\ &
    -G \leq u_{y,e} \leq G
\end{aligned}
$$

# Chaser Optimal Control derivations: 
To simplify the constraints so the problem can be converted into a BVP, here we chose a simple positional state space model. 
The Hamiltonian is then linear with respect to the inputs, and so the problem is in the nature of Bang-Bang control. 
Since the Bang-Bang control can cover the constraints, and we can let the Hamiltonian simply be:
$$
H = \lambda^T(Ax+Bu)
$$

We thus have the State dynamics: 
$$
\dot{x} = \frac{\partial H}{\partial \lambda} = Ax+Bu
$$

The co-state dynamics: 
$$
\dot{\lambda}=-\frac{\partial H}{\partial x} = \begin{bmatrix}
    0 \\ 
    0
\end{bmatrix}
$$

Control optimality: 
$$
\frac{\partial H}{\partial u} = \begin{bmatrix}
    \lambda_1 \\ 
    \lambda_2
\end{bmatrix} \rightarrow u = \begin{bmatrix}
    -\text{sign}(\lambda_1)V \\ 
    -\text{sign}(\lambda_2)V
\end{bmatrix}
$$
