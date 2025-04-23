%% Parameters
Ts = 0.1;  % Time step
m = 1;     % Mass
time_frames = 60;
capture_radius = 3;  % Distance at which pursuer catches evader

%% init state
x_evader = zeros(4, time_frames);
x_pursuer = zeros(4, time_frames);
x_evader(:, 1) = [10, 20, 3, 0];  % Initial state [x, y, vx, vy]
x_pursuer(:, 1) = [50, 50, -3, -3];

a_evader = [0, -5];  % Fixed acceleration for evader

%% System dynamics matrices
A = [1, 0, Ts, 0;
    0, 1, 0, Ts;
    0, 0, 1, 0;
    0, 0, 0, 1];
    
B = zeros(4,2);
B(3,1) = Ts/m; 
B(4,2) = Ts/m;

%% MPC Parameters
Q = diag([10, 10, 1, 1]); 
R = diag([1, 1]);        
N = 10;                  

% Constraint matrices
E = [eye(2*N); -eye(2*N)];  
W = ones(4*N,1) * 10;        

%% EKF parameters
P = eye(4);              
Qk = 0.05 * eye(4);    
Rk = 0.5 * eye(2);       

%% Game loop
for t = 1:time_frames-1
    % Current states
    evader_current = x_evader(:, t);
    pursuer_current = x_pursuer(:, t);
    
    % measurements 
    z_evader = evader_current(1:2) + 0.2 * randn(2, 1); 
    z_pursuer = pursuer_current(1:2) + 0.2 * randn(2, 1);
    
    % Update evader state using fixed acceleration
    u_evader = a_evader';
    x_evader(:, t+1) = A * evader_current + B * u_evader;
    
    % Estimate evader's future trajectory for pursuer planning
    evader_pred = x_evader(:, t+1);
    evader_future = zeros(4, N);
    evader_future(:, 1) = evader_pred;
    
    % Predict evader's future positions based on constant acceleration
    for i = 2:N
        evader_future(:, i) = A * evader_future(:, i-1) + B * u_evader;
    end
    
    % Determine optimal interception point for pursuer
    % Choose the point that minimizes the distance after N steps
    optimal_target = evader_future(:, N);
    
    % Check if capture is possible within prediction horizon
    capture_possible = false;
    if capture_possible == false
        for step = 1:N
            distance = norm(evader_future(1:2, step) - pursuer_current(1:2));  
            time_to_reach = step * Ts;
            if distance / time_to_reach < 10  % If pursuer can reach evader position
                optimal_target = evader_future(:, step);
                capture_possible = true;
                break;
            end
        end
    end
    % If capture doesn't seem possible within horizon, aim for final predicted position
    if ~capture_possible
        optimal_target = evader_future(:, N);
    end
    
    % Use MPC to compute optimal control for pursuer
    [u_pursuer, x_pred] = mpc(pursuer_current, optimal_target, N, A, B, Q, R, E, W);

    % Update pursuer state
    x_pursuer(:, t+1) = A * pursuer_current + B * u_pursuer(:, 1);
    
    % Check for capture
    distance = norm(x_evader(1:2, t+1) - x_pursuer(1:2, t+1));
    if distance < capture_radius
        disp(['Evader captured at time step ', num2str(t+1), '!']);
        break;
    end
end

% Visualize the game
figure;
hold on;
plot(x_evader(1, 1:t+1), x_evader(2, 1:t+1), 'b-', 'LineWidth', 2);
plot(x_pursuer(1, 1:t+1), x_pursuer(2, 1:t+1), 'r-', 'LineWidth', 2);
plot(x_evader(1, 1), x_evader(2, 1), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
plot(x_pursuer(1, 1), x_pursuer(2, 1), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(x_evader(1, t+1), x_evader(2, t+1), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
plot(x_pursuer(1, t+1), x_pursuer(2, t+1), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

% Draw capture if it happened
if distance < capture_radius
    disp('captured')
    plot(x_evader(1, t+1), x_evader(2, t+1), 'ko', 'MarkerSize', 15);
end

title('Pursuer-Evader Game');
xlabel('X Position');
ylabel('Y Position');
legend('Evader Path', 'Pursuer Path', 'Evader Start', 'Pursuer Start', 'Evader End', 'Pursuer End');
grid on;
axis equal;

function [uMPC, xMPC] = mpc(x0, target, N, A, B, Q, R, E, W)

    % mpc
    P = Q;
    
    G = zeros(N*4, N*2);
    for i = 1:N
        for j = 1:N
            if i >= j
                G((i-1)*4+1:i*4, (j-1)*2+1:j*2) = A^(i-j) * B;
            end
        end
    end
    Qbar = blkdiag(Q, Q, Q, Q, Q, Q, Q, Q, Q, P);
    Rbar = blkdiag(R, R, R, R, R, R, R, R, R, R);
    L = G'*Qbar*G + Rbar;
    epsilon = 1e-6;
    L = L + epsilon * eye(size(L));
    H = [A;
         A^2;
         A^3;
         A^4;
         A^5;
         A^6;
         A^7;
         A^8;
         A^9;
         A^10;]; 
    F = G' * Qbar * H;
    % A lower triangular matrix Lo, Lo*Lo'=L
    Lo = chol(L, 'lower');
    
    % Inverse of Lo
    Linv = Lo\eye(size(L,1));
    
    % Define all inequality constraints as inactive because of the solver
    iA = false(size(W));
    
    % Define a default option set for mpcsolver
    opt = mpcActiveSetOptions;
    opt.IntegrityChecks = false;
    t_constrained =  0:50;
    uMPC = zeros(2, length(t_constrained));
    xMPC = zeros(4, length(t_constrained));
    xMPC(:,1) = x0;

    x = x0;

    y = x; %H_m * x;
    for ct = t_constrained
        x_hat = y;  % x_hat = ekf(y)
        [u, status, iA] = mpcActiveSetSolver(L, F*(x_hat-target), E, W, [], zeros(0, 1), iA, opt);
        
        % Save the first input of the input sequence
        uMPC(:, ct+1) = u(1:2);
        
        % Apply the first input to the system
        x = A*x + B*u(1:2);
        xMPC(:, ct+1) = x;
        % Apply the first input to the system
        x = A*x+B*u(1:2);
        y = x; %H_m*x+noise;
        
    end
    % figure;
    % subplot(2, 1, 1)
    % plot([t_constrained],xMPC(1,:), [t_constrained],xMPC(2,:))
    % hold on;
    % yline(target(1), '--', 'LineWidth', 1.5);
    % yline(target(2), '--', 'LineWidth', 1.5);
    % title('Histories of the Quadrotor State')
    % xlabel('Time (s)')
    % ylabel('x(t)')
    % legend('x1 horizontal position','x2 vertical position')
    % 
    % subplot(2, 1, 2)
    % plot([t_constrained],uMPC(1,:));
    % hold on;
    % plot([t_constrained],uMPC(2,:));
    % title('Histories of the Control Inputs')
    % xlabel('Time (s)')
    % ylabel('u(t)')
    % legend('u1','u2')

end


function [x_kplus_hat, P] = ekf_func(z_k, x_kminus, P, Qk, Rk, acc_x_k, acc_y_k, dt_k)

    % Predict
    x_k_pred = x_kminus;
    x_k_pred(1) = x_k_pred(1) + x_k_pred(3)*dt_k + 0.5*acc_x_k*dt_k^2;
    x_k_pred(2) = x_k_pred(2) + x_k_pred(4)*dt_k + 0.5*acc_y_k*dt_k^2;
    x_k_pred(3) = x_k_pred(3) + acc_x_k*dt_k;
    x_k_pred(4) = x_k_pred(4) + acc_y_k*dt_k;

    A = eye(4);
    A(1,3) = dt_k;
    A(2,4) = dt_k;
    
    P = A*P*A' + Qk;
    % Measurement update using GPS
    H = [1 0 0 0; 0 1 0 0];
    y_res = z_k - H*x_k_pred;
    S = H*P*H' + Rk;
    K = P*H'/S;
    
    %Output
    x_kplus_hat = x_k_pred + K*y_res;
    P = (eye(4) - K*H) * P;

end