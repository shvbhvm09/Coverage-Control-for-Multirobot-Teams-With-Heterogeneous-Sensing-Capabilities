% Coverage control using Heterogeneous Lloyd's Algorithm.
% Shubham Daheriya
% 10/2024

%% Experiment Constants
iterations = 2000;
N = 7;
sigma = 0.7;
kappa = 1.0;

% Define heterogeneous sensing centers
rng('default');
for i = 1:N
    density_params(i).mu = [-1.5 + 3*rand(); -1 + 2*rand()];
    density_params(i).beta = 1;
end

%%%%%%%%%%%%%%%%%%%%% For the 1st edge case - centroid at the same place

% % Overlapping density centers for first 3 robots
% density_params(1).mu = [-0.5; 0.0];
% density_params(2).mu = [-0.5; 0.0];
% density_params(3).mu = [-0.5; 0.0];
% 
% % Spread out the rest
% density_params(4).mu = [1.0; 0.8];
% density_params(5).mu = [-1.2; -0.7];
% density_params(6).mu = [0.3; -0.9];
% density_params(7).mu = [1.4; -0.4];
% 
% % Set beta values
% for i = 1:N
%     density_params(i).beta = 1;
% end

% %%%%%%%%%%%%%%% For the 2nd edge case - Let's now test what happens if one robot is way more important (e.g., β = 5) and others are less (β = 0.5).
% 
% density_params(1).mu = [0; 0];           % Robot with β = 5 (high priority)
% density_params(2).mu = [1.0; 0.8];
% density_params(3).mu = [-1.2; -0.7];
% density_params(4).mu = [0.3; -0.9];
% density_params(5).mu = [1.4; -0.4];
% density_params(6).mu = [-1.3; 0.6];
% density_params(7).mu = [0.8; -0.8];
% 
% density_params(1).beta = 5;     % Highly important
% for i = 2:7
%     density_params(i).beta = 0.5;  % Lower importance
% end

%%%%%%%%%%%%%%%%%%%%%%%% For 3rd edge Case 3: Subgroups with identical sensors

% for i = 1:2:N
%     density_params(i).mu = [-1; 0]; % Group 1
%     density_params(i).beta = 1;
% end
% for i = 2:2:N
%     density_params(i).mu = [1; 0]; % Group 2
%     density_params(i).beta = 1;
% end




x_init = generate_initial_conditions(N,'Width',1.1,'Height',1.1,'Spacing', 0.35);
x_init = x_init - [min(x_init(1,:)) - (-1.6 + 0.2);min(x_init(2,:)) - (-1 + 0.2);0];
r = Robotarium('NumberOfRobots', N, 'ShowFigure', true,'InitialConditions',x_init);

%Initialize velocity vector
dxi = zeros(2, N);

%%%%%%%%%%%%%%%%%%%% CHANGES FOR TET CASE #3 OBSTACLE PRESENCE%%%%%%%%%%%%
% %Boundary
% crs = [r.boundaries(1), r.boundaries(3);
%        r.boundaries(1), r.boundaries(4);
%        r.boundaries(2), r.boundaries(4);
%        r.boundaries(2), r.boundaries(3)];

% Outer boundary (rectangle)
outer = [ -1.6, -1.0;
           -1.6,  1.0;
            1.6,  1.0;
            1.6, -1.0 ];

% Obstacle (inner hole — a square in the middle)
inner = [ -0.4, -0.4;
          -0.4,  0.4;
           0.4,  0.4;
           0.4, -0.4 ];

% Combine using polyshape with a hole
domain_shape = polyshape(outer(:,1), outer(:,2));
hole_shape   = polyshape(inner(:,1), inner(:,2));
domain       = subtract(domain_shape, hole_shape);

% Extract boundary from resulting shape for VoronoiBounded
[xb, yb] = boundary(domain);
crs = [xb, yb];


% Unicycle dynamics & safety
[~, uni_to_si_states] = create_si_to_uni_mapping();
si_to_uni_dyn = create_si_to_uni_dynamics();
uni_barrier_cert_boundary = create_uni_barrier_certificate_with_boundary();

% Plotting Setup
marker_size = determine_marker_size(r, 0.08);
x = r.get_poses();

verCellHandle = zeros(N,1);
cellColors = cool(N);
for i = 1:N
    verCellHandle(i)  = patch(x(1,i),x(2,i),cellColors(i,:),'FaceAlpha', 0.3);
    hold on
end
pathHandle = zeros(N,1);
for i = 1:N
    pathHandle(i)  = plot(x(1,i),x(2,i),'-.','color',cellColors(i,:)*.9, 'LineWidth',4);
end
centroidHandle = plot(x(1,:),x(2,:),'+','MarkerSize',marker_size, 'LineWidth',2, 'Color', 'k');

for i = 1:N
    xD = [get(pathHandle(i),'XData'),x(1,i)];
    yD = [get(pathHandle(i),'YData'),x(2,i)];
    set(pathHandle(i),'XData',xD,'YData',yD);
end

% Visualize density centers
for i = 1:N
    plot(density_params(i).mu(1), density_params(i).mu(2), 'x', 'Color', cellColors(i,:), 'MarkerSize', 10, 'LineWidth', 2);
end

r.step();

%% Main Loop
cost_history = zeros(iterations, 1); % Initialize cost tracking

for t = 1:iterations
    x = r.get_poses();
    Px = x(1,:)';
    Py = x(2,:)';
    robot_positions = x(1:2,:);

    % Compute and store cost
    cost_history(t) = computeHeterogeneousCost(Px, Py, crs, robot_positions, density_params, sigma);

    dxi = heterogeneousLloydsAlgorithm(Px, Py, crs, robot_positions, density_params, sigma, kappa);

    norms = arrayfun(@(x) norm(dxi(:, x)), 1:N);
    threshold = 3/4*r.max_linear_velocity;
    to_thresh = norms > threshold;
    dxi(:, to_thresh) = threshold*dxi(:, to_thresh)./norms(to_thresh);

    dxu = si_to_uni_dyn(dxi, x);
    dxu = uni_barrier_cert_boundary(dxu, x);
    r.set_velocities(1:N, dxu);

    % Update plots and Voronoi cells (existing code)
    for i = 1:N
        xD = [get(pathHandle(i),'XData'),x(1,i)];
        yD = [get(pathHandle(i),'YData'),x(2,i)];
        set(pathHandle(i),'XData',xD,'YData',yD);
    end
    [v, c] = VoronoiBounded(Px, Py, crs);
    for i = 1:N
        if ~isempty(c{i})
            set(verCellHandle(i), 'XData', v(c{i},1), 'YData', v(c{i},2));
        end
    end
    set(centroidHandle,'XData',Px,'YData',Py);
    r.step();
end

% Plot cost history
figure;
plot(1:iterations, cost_history, 'LineWidth', 2);
xlabel('Iteration', 'Interpreter', 'latex');
title('Heterogeneous Locational Cost Over Time', 'Interpreter', 'latex');
ylabel('$\mathcal{H}_{\mathrm{het}}$', 'Interpreter', 'latex');
title('Heterogeneous Locational Cost Over Time');
grid on;

r.debug();

%% Helper Functions
function marker_size = determine_marker_size(robotarium_instance, marker_size_meters)
curunits = get(robotarium_instance.figure_handle, 'Units');
set(robotarium_instance.figure_handle, 'Units', 'Points');
cursize = get(robotarium_instance.figure_handle, 'Position');
set(robotarium_instance.figure_handle, 'Units', curunits);
marker_ratio = (marker_size_meters)/(robotarium_instance.boundaries(2) - robotarium_instance.boundaries(1));
marker_size = cursize(3) * marker_ratio;
end

function dxi = heterogeneousLloydsAlgorithm(Px, Py, crs, robot_positions, density_params, sigma, kappa)
N = numel(Px);
dxi = zeros(2,N);
[v, c] = VoronoiBounded(Px, Py, crs);

for i = 1:N
    mu_i = density_params(i).mu;
    beta_i = density_params(i).beta;
    Sigma = 0.1*eye(2);
    invSigma = inv(Sigma);
    phi_i = @(x,y) beta_i .* exp(-0.5 * (([x;y] - mu_i)' * invSigma * ([x;y] - mu_i)));

    cell_coords = v(c{i},:);
    [m_i, c_i] = computeMassCentroid(cell_coords, phi_i);
    [M_i, C_i] = computeMassCentroid(crs, phi_i);

    boundary_term = zeros(2,1);
    for j = 1:N
        if j == i || isempty(intersect(c{i}, c{j})), continue; end
        shared_idx = intersect(c{i}, c{j});
        boundary_pts = v(shared_idx, :);
        if size(boundary_pts,1) < 2, continue; end

        q_line = linspacePoints(boundary_pts(1,:), boundary_pts(end,:), 10);
        mu_ij = 0; rho_ij = zeros(2,1);
        mu_ji = 0; rho_ji = zeros(2,1);

        for k = 1:size(q_line,1)
            q = q_line(k,:)';
            dist2 = norm(q - robot_positions(:,i))^2;
            val_i = beta_i * exp(-0.5 * ((q - mu_i)' * invSigma * (q - mu_i)));
            mu_ij = mu_ij + dist2 * val_i;
            rho_ij = rho_ij + q * dist2 * val_i;

            mu_j = density_params(j).beta * exp(-0.5 * ((q - density_params(j).mu)' * invSigma * (q - density_params(j).mu)));
            mu_ji = mu_ji + dist2 * mu_j;
            rho_ji = rho_ji + q * dist2 * mu_j;
        end

        if mu_ij > 1e-6, rho_ij = rho_ij / mu_ij; end
        if mu_ji > 1e-6, rho_ji = rho_ji / mu_ji; end

        boundary_term = boundary_term + mu_ij*(rho_ij - robot_positions(:,i)) - mu_ji*(rho_ji - robot_positions(:,i));
    end

    term1 = 2*sigma*m_i*(robot_positions(:,i) - c_i);
    term2 = 2*(1 - sigma)*M_i*(robot_positions(:,i) - C_i);
    term3 = sigma * boundary_term;
    dxi(:,i) = -kappa * (term1 + term2 + term3);
end
end

function [mass, centroid] = computeMassCentroid(polygon, phi)
    poly = polyshape(polygon);
    [xg, yg] = meshgrid(linspace(min(polygon(:,1)), max(polygon(:,1)), 40), ...
                        linspace(min(polygon(:,2)), max(polygon(:,2)), 40));
    % Flatten for compatibility
    inside = isinterior(poly, xg(:), yg(:));
    x_in = xg(:); x_in = x_in(inside);
    y_in = yg(:); y_in = y_in(inside);
    
    val = arrayfun(phi, x_in, y_in);

    mass = sum(val);
    if mass > 1e-6
        cx = sum(x_in .* val) / mass;
        cy = sum(y_in .* val) / mass;
    else
        cx = mean(polygon(:,1));
        cy = mean(polygon(:,2));
    end
    centroid = [cx; cy];
end


function pts = linspacePoints(p1, p2, num_pts)
x = linspace(p1(1), p2(1), num_pts);
y = linspace(p1(2), p2(2), num_pts);
pts = [x', y'];
end

function H_het = computeHeterogeneousCost(Px, Py, crs, robot_positions, density_params, sigma)
    N = numel(Px);
    [v, c] = VoronoiBounded(Px, Py, crs);
    H_C = 0; % Coordination term
    H_O = 0; % Domain objectives term

    for i = 1:N
        mu_i = density_params(i).mu;
        beta_i = density_params(i).beta;
        Sigma = 0.1*eye(2);
        invSigma = inv(Sigma);
        phi_i = @(x,y) beta_i .* exp(-0.5 * (([x;y] - mu_i)' * invSigma * ([x;y] - mu_i)));

        % Coordination term (H_C): Integral over Voronoi cell
        cell_coords = v(c{i},:);
        [m_i, ~] = computeMassCentroid(cell_coords, phi_i);
        H_C = H_C + m_i;

        % Domain objectives term (H_O): Integral over entire domain
        [M_i, ~] = computeMassCentroid(crs, phi_i);
        H_O = H_O + M_i;
    end

    % Heterogeneous cost (Equation 7)
    H_het = sigma * H_C + (1 - sigma) * H_O;
end