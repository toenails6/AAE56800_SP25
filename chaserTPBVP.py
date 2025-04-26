import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_bvp


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


if __name__ == "__main__":
    # Test parameters.
    x_c_0 = np.array([-1, -1])
    x_e_0 = np.array([1, 0, 0, 1])
    timeStep = 4
    maxSpeed = 1E-1
    areaBnds = np.array([-50, 50, -50, 50])
    areaBnds[1] = -0.8

    retval = chaserTPBVP(
        x_c_0, x_e_0, maxSpeed, areaBnds, timeStep)

    print(retval)
