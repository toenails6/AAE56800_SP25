% Packaged TPBVP chaser function. 
% x_c_0 = [x_c; y_c]; Chaser initial states. 
% x_e_0 = [x_e; y_e; v_x_e; v_y_e]; Evader initial states. 
% maxSpeed: maximum speed. 
% areaBnds = [x_l; x_u; y_l; y_u]; 
% timeStep: time span in the simulation that this function covers. 
function retval = chaserTPBVP(x_c_0, x_e_0, maxSpeed, areaBnds, timeStep)
    % Evader position estimation. 
    x_e = x_e_0(1:2)+timeStep*x_e_0(3:4); 

    % TPBVP solver time span. 
    tspan = 0 : 1E-1 : timeStep; 

    % Initial guess. 
    guess = [x_e; 0; 0]; 
    solGuess = bvpinit(tspan, guess); 
    sol = bvp4c( ...
        @(t, y) eqns(t, y, maxSpeed, areaBnds), ...
        @(y_a, y_b) boundaryConditions(y_a, y_b, x_c_0, x_e), ...
        solGuess); 

    % Solutions. 
    y = sol.y; 
    retval = y(1:2, end); 

    % Differential equations. 
    function dydt = eqns(~, y, maxSpeed, areaBnds)
        % Equivalence threshold. 
        eps = 1E-3; 
    
        % Optimal Control. 
        u = (abs(y(3:4))>eps) .* -sign(y(3:4))*maxSpeed; 
        
        % Position and velocity constraints. 
        if y(1) < areaBnds(1)
            u(1) = maxSpeed; 
        elseif areaBnds(2) < y(1)
            u(1) = -maxSpeed; 
        end
        if y(2) < areaBnds(3)
            u(2) = maxSpeed; 
        elseif areaBnds(4) < y(2)
            u(2) = -maxSpeed; 
        end
    
        A = [zeros(2)]; 
        B = [eye(2)]; 
    
        dydt = [ ...
            A*y(1:2)+B*u; ...
            zeros(2, 1)]; 
    end
    
    % Boundary conditions. 
    function retval = boundaryConditions(y_a, y_b, x_c_0, x_e)
        retval = [ ...
            y_a(1:2)-x_c_0; ...
            y_b(3:4)-(y_b(1:2)-x_e)]; 
    end
end
