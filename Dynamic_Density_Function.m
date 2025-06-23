% Generalized Coverage Control with Time-Varying Density Functions
% Based on: "Generalized Coverage Control for Time-Varying Density Functions"
% by James Kennedy, Airlie Chapman and Peter M. Dower (ECC 2019)
clc;
clear;
% Shubham Daheriya
% 05/2025

%% Experiment Constants
iterations = 2000;
dt = 0.033; % Time step (approx. same as Robotarium step time)


% Create figure for metrics
metrics_fig = figure('Name', 'Performance Metrics');
ax1 = subplot(3,1,1); title('Coverage Cost (Hv)'); hold on;
ax2 = subplot(3,1,2); title('Tracking Error'); hold on;
ax3 = subplot(3,1,3); title('Control Effort'); hold on;


%% Set up the Robotarium object
N = 7;
x_init = generate_initial_conditions(N,'Width',1.1,'Height',1.1,'Spacing', 0.35);
offset_x = min(x_init(1,:)) - (-1.6 + 0.2);
offset_y = min(x_init(2,:)) - (-1 + 0.2);
x_init = x_init - [offset_x * ones(1, N); offset_y * ones(1, N); zeros(1, N)];
r = Robotarium('NumberOfRobots', N, 'ShowFigure', true,'InitialConditions',x_init);


% Metrics collection
metrics = struct();
metrics.time = zeros(1, iterations);
metrics.Hv = zeros(1, iterations); % Coverage cost
metrics.tracking_error = zeros(1, iterations); % Distance to density centroid
metrics.control_effort = zeros(1, iterations); % Total control input
metrics.voronoi_areas = zeros(N, iterations); % Area of each Voronoi cell
metrics.voronoi_masses = zeros(N, iterations); % Mass of each Voronoi cell
metrics.centroid_distances = zeros(N, iterations); % Distance to centroid

% Initialize velocity vector
dxi = zeros(2, N);

% Boundary
crs = [r.boundaries(1), r.boundaries(3);
       r.boundaries(1), r.boundaries(4);
       r.boundaries(2), r.boundaries(4);
       r.boundaries(2), r.boundaries(3)];

%% Control parameters (from paper)
k = 1.0; % Basic convergence gain
k1 = 5.0; % Additional gain for time-varying compensation
C2 = 0.1; % Convergence radius threshold

%% Density function parameters (example from paper)
a = 1/8; % Speed parameter for moving density
density_type = 1; % 1 = moving Gaussian, 2 = mass-varying, 3 = elliptical

%% Grab tools for dynamics conversion
[~, uni_to_si_states] = create_si_to_uni_mapping();
si_to_uni_dyn = create_si_to_uni_dynamics();
uni_barrier_cert_boundary = create_uni_barrier_certificate_with_boundary();

%% Plotting Setup
marker_size = determine_marker_size(r, 0.08);
x = r.get_poses();

verCellHandle = zeros(N,1);
cellColors = cool(N);
for i = 1:N
    verCellHandle(i) = patch(x(1,i),x(2,i),cellColors(i,:),'FaceAlpha', 0.3);
    hold on
end

pathHandle = zeros(N,1);      
for i = 1:N
    pathHandle(i) = plot(x(1,i),x(2,i),'-.','color',cellColors(i,:)*.9, 'LineWidth',4);
end
centroidHandle = plot(x(1,:),x(2,:),'+','MarkerSize',marker_size, 'LineWidth',2, 'Color', 'k');

for i = 1:N
    xD = [get(pathHandle(i),'XData'),x(1,i)];
    yD = [get(pathHandle(i),'YData'),x(2,i)];
    set(pathHandle(i),'XData',xD,'YData',yD);
end

r.step();

%% Initialize variables for the algorithm
prev_phi = zeros(N,1);
prev_mass = zeros(N,1);
prev_centroids = zeros(2,N);

%% Main Loop
for t = 1:iterations
    % Retrieve poses
    x = r.get_poses();
    xi = uni_to_si_states(x);
    
    % Current time for time-varying density
    current_time = t * dt;
    
    %% Compute Voronoi partitions
    [Px, Py, V, C] = generalizedVoronoi(x(1,:)',x(2,:)', crs, verCellHandle);
    
    %% Compute density function and its time derivative
    [phi, dphi_dt] = computeDensityFunction(Px, Py, current_time, density_type, a);
    
    %% Compute mass and centroids (from paper equations)
    mass = zeros(N,1);
    centroids = zeros(2,N);
    dmass_dt = zeros(N,1);
    dcentroids_dt = zeros(2,N);
    
    for i = 1:N
        % Current cell vertices
        v_cell = V(C{i},:);
        
        % Compute mass (equation 4)
        mass(i) = polyintegral(@(q) phi(q), v_cell);
        
        % Compute centroid (equation 3)
        if mass(i) > eps
            cx = polyintegral(@(q) q(1)*phi(q), v_cell) / mass(i);
            cy = polyintegral(@(q) q(2)*phi(q), v_cell) / mass(i);
            centroids(:,i) = [cx; cy];
        else
            centroids(:,i) = [Px(i); Py(i)];
        end
        
    % Store time
    metrics.time(t) = current_time;
    
    % Compute and store coverage cost (equation 2 from paper)
    Hv = 0;
    for i = 1:N
        v_cell = V(C{i},:);
        Hv = Hv + polyintegral(@(q) norm(q-[Px(i); Py(i)])^2 * phi(q), v_cell);
    end
    metrics.Hv(t) = Hv;
    
    % Compute tracking error (distance to density centroid)
    if density_type == 1 % For moving Gaussian
        x0 = 0.5*cos(a*pi*current_time);
        y0 = 0.5*sin(a*pi*current_time);
        density_centroid = [x0; y0];
        metrics.tracking_error(t) = norm(mean(x(1:2,:), 2) - density_centroid);
    end
    
    % Compute control effort
    metrics.control_effort(t) = sum(vecnorm(dxi));
    
    % Store Voronoi cell metrics
    for i = 1:N
        v_cell = V(C{i},:);
        metrics.voronoi_areas(i,t) = polyarea(v_cell(:,1), v_cell(:,2));
        metrics.voronoi_masses(i,t) = mass(i);
        metrics.centroid_distances(i,t) = norm(x(1:2,i) - centroids(:,i));
    end
    
    % Update metrics plots every 50 iterations for efficiency
    if mod(t,50) == 0 || t == iterations
        figure(metrics_fig);
        
        % Coverage Cost
        plot(ax1, metrics.time(1:t), metrics.Hv(1:t), 'b-', 'LineWidth', 1.5);
        xlabel(ax1, 'Time (s)'); ylabel(ax1, 'Hv');
        grid(ax1, 'on');
        
        % Tracking Error
        if density_type == 1
            plot(ax2, metrics.time(1:t), metrics.tracking_error(1:t), 'r-', 'LineWidth', 1.5);
            xlabel(ax2, 'Time (s)'); ylabel(ax2, 'Error');
            grid(ax2, 'on');
        end
        
        % Control Effort
        plot(ax3, metrics.time(1:t), metrics.control_effort(1:t), 'g-', 'LineWidth', 1.5);
        xlabel(ax3, 'Time (s)'); ylabel(ax3, 'Control Effort');
        grid(ax3, 'on');
        
        drawnow;
    end

        % Compute time derivatives (if not first step)
        if t > 1
            dt_actual = dt; % Could use actual time difference if available
            
            % Time derivative of mass (dm_i/dt)
            dmass_dt(i) = polyintegral(@(q) dphi_dt(q), v_cell);
            
            % Time derivative of centroid (dc_i/dt) - from paper
            if mass(i) > eps
                term1 = polyintegral(@(q) [q(1); q(2)]*dphi_dt(q), v_cell);
                dcentroids_dt(:,i) = (term1 - dmass_dt(i)*centroids(:,i)) / mass(i);
            end
        end
    end
    
    %% Generalized Control Law (equation 19 from paper)
    for i = 1:N
        if mass(i) > eps
            % Additional term from paper: integral of ||q-c_i||^2 over V_i
            integral_term = polyintegral(@(q) norm(q-centroids(:,i))^2, v_cell);
            
            % Control law (equation 19)
            u_i = dcentroids_dt(:,i) - (k1/mass(i)*integral_term + k + dmass_dt(i)/mass(i)) * (xi(1:2,i) - centroids(:,i));
        else
            % Fallback to standard Lloyd's if mass is zero
            u_i = -k * (xi(1:2,i) - [Px(i); Py(i)]);
        end
        
        dxi(:,i) = u_i;
    end
    
    %% Store current values for next iteration
    prev_phi = phi;
    prev_mass = mass;
    prev_centroids = centroids;
    
    %% Avoid actuator errors
    norms = arrayfun(@(x) norm(dxi(:, x)), 1:N);
    threshold = 3/4*r.max_linear_velocity;
    to_thresh = norms > threshold;
    dxi(:, to_thresh) = threshold*dxi(:, to_thresh)./norms(to_thresh);
    
    %% Use barrier certificate and convert to unicycle dynamics
    dxu = si_to_uni_dyn(dxi, x);
    dxu = uni_barrier_cert_boundary(dxu, x);
    
    %% Send velocities to agents
    r.set_velocities(1:N, dxu);
    
    %% Update Plot Handles
    for i = 1:N
       xD = [get(pathHandle(i),'XData'),x(1,i)];
       yD = [get(pathHandle(i),'YData'),x(2,i)];
       set(pathHandle(i),'XData',xD,'YData',yD);
    end
    
    set(centroidHandle,'XData',centroids(1,:),'YData',centroids(2,:));
     
    % Iterate experiment
    r.step();
end

% Final metrics analysis and visualization
figure('Name', 'Final Metrics Analysis');

% Voronoi cell areas over time
subplot(2,2,1);
plot(metrics.time, metrics.voronoi_areas);
title('Voronoi Cell Areas Over Time');
xlabel('Time (s)'); ylabel('Area');
grid on;

% Voronoi cell masses over time
subplot(2,2,2);
plot(metrics.time, metrics.voronoi_masses);
title('Voronoi Cell Masses Over Time');
xlabel('Time (s)'); ylabel('Mass');
grid on;

% Distance to centroids over time
subplot(2,2,3);
plot(metrics.time, metrics.centroid_distances);
title('Distance to Centroids Over Time');
xlabel('Time (s)'); ylabel('Distance');
grid on;

% Coverage cost vs control effort
subplot(2,2,4);
scatter(metrics.control_effort, metrics.Hv, 10, metrics.time, 'filled');
colorbar; title('Coverage Cost vs Control Effort');
xlabel('Control Effort'); ylabel('Coverage Cost (Hv)');
grid on;

% Display summary statistics
fprintf('\n=== Performance Metrics Summary ===\n');
fprintf('Final Coverage Cost (Hv): %.4f\n', metrics.Hv(end));
fprintf('Average Control Effort: %.4f\n', mean(metrics.control_effort));
fprintf('Maximum Tracking Error: %.4f\n', max(metrics.tracking_error));
fprintf('Average Distance to Centroids: %.4f\n', mean(metrics.centroid_distances(:)));
fprintf('Convergence Time (to C2=%.2f): %.2f s\n', C2, metrics.time(find(metrics.centroid_distances(:) < C2, 1)));

r.debug();

%% Helper Functions

function marker_size = determine_marker_size(robotarium_instance, marker_size_meters)
    curunits = get(robotarium_instance.figure_handle, 'Units');
    set(robotarium_instance.figure_handle, 'Units', 'Points');
    cursize = get(robotarium_instance.figure_handle, 'Position');
    set(robotarium_instance.figure_handle, 'Units', curunits);
    marker_ratio = (marker_size_meters)/(robotarium_instance.boundaries(2) -...
        robotarium_instance.boundaries(1));
    marker_size = cursize(3) * marker_ratio;
end

function [Px, Py, V, C] = generalizedVoronoi(Px,Py, crs, verCellHandle)
    [V,C] = VoronoiBounded(Px,Py,crs);
    
    for i = 1:numel(C)
        set(verCellHandle(i), 'XData',V(C{i},1),'YData',V(C{i},2));
    end
end

function [V,C]=VoronoiBounded(x,y, crs)
    bnd=[min(x) max(x) min(y) max(y)];
    if nargin < 3
        crs=double([bnd(1) bnd(4);bnd(2) bnd(4);bnd(2) bnd(3);bnd(1) bnd(3);bnd(1) bnd(4)]);
    end

    rgx = max(crs(:,1))-min(crs(:,1));
    rgy = max(crs(:,2))-min(crs(:,2));
    rg = max(rgx,rgy);
    midx = (max(crs(:,1))+min(crs(:,1)))/2;
    midy = (max(crs(:,2))+min(crs(:,2)))/2;

    xA = [x; midx + [0;0;-5*rg;+5*rg]];
    yA = [y; midy + [-5*rg;+5*rg;0;0]];

    [vi,ci]=voronoin([xA,yA]);

    C = ci(1:end-4);
    V = vi;

    for ij=1:length(C)
        [X2, Y2] = poly2cw_custom(V(C{ij},1),V(C{ij},2));
        tempA = polyshape(crs(:,1), crs(:,2),'Simplify',false);
        tempB = polyshape(X2, Y2,'Simplify',false);
        tempC = intersect(tempA,tempB);
        [xb, yb] = boundary(tempC);
        
        ix=nan(1,length(xb));
        for il=1:length(xb)
            if any(V(:,1)==xb(il)) && any(V(:,2)==yb(il))
                ix1=find(V(:,1)==xb(il));
                ix2=find(V(:,2)==yb(il));
                for ib=1:length(ix1)
                    if any(ix1(ib)==ix2)
                        ix(il)=ix1(ib);
                    end
                end
                if isnan(ix(il))==1
                    lv=length(V);
                    V(lv+1,1)=xb(il);
                    V(lv+1,2)=yb(il);
                    ix(il)=lv+1;
                end
            else
                lv=length(V);
                    V(lv+1,1)=xb(il);
                    V(lv+1,2)=yb(il);
                    ix(il)=lv+1;
            end
        end
        C{ij}=ix;
    end
end

function [ordered_x, ordered_y] = poly2cw_custom(x,y)
    cx = mean(x);
    cy = mean(y);
    a = atan2(y-cy, x-cx);
    [~, order] = sort(a);
    ordered_x = x(order);
    ordered_y = y(order);
end

function [phi, dphi_dt] = computeDensityFunction(x, y, t, type, a)
    % Implement the time-varying density functions from the paper
    % Returns both the density and its time derivative
    
    switch type
        case 1 % Moving Gaussian (equation from Fig. 4)
            x0 = 0.5*cos(a*pi*t);
            y0 = 0.5*sin(a*pi*t);
            
            % Density function
            phi = @(q) 10*exp(-5*(q(1)-x0)^2 - 5*(q(2)-y0)^2) + 0.1;
            
            % Time derivative of density
            dx0_dt = -0.5*a*pi*sin(a*pi*t);
            dy0_dt = 0.5*a*pi*cos(a*pi*t);
            dphi_dt = @(q) phi(q) * (10*(q(1)-x0)*dx0_dt + 10*(q(2)-y0)*dy0_dt);
            
        case 2 % Mass-varying Gaussian (equation from Fig. 9)
            % Density function
            phi = @(q) 5*sin(0.1*pi*t)*exp(-(q(1)/0.4)^2 - (q(2)/0.4)^2) + 0.1;
            
            % Time derivative
            dphi_dt = @(q) 0.5*pi*cos(0.1*pi*t)*exp(-(q(1)/0.4)^2 - (q(2)/0.4)^2);
            
        case 3 % Elliptical configuration (equation from Fig. 12)
            y_offset = 0.3*sin(pi*t/8);
            % Density function
            phi = @(q) exp(-500*(-0.3^2 + 1.4*q(1)^2 + 0.6*(q(2)-y_offset)^2)^2) + 0.1;
            
            % Time derivative
            dy_offset_dt = 0.3*pi/8*cos(pi*t/8);
            dphi_dt = @(q) phi(q) * (-1000*(-0.3^2 + 1.4*q(1)^2 + 0.6*(q(2)-y_offset)^2)) * ...
                         (1.2*(q(2)-y_offset)*(-dy_offset_dt));

            
        otherwise % Uniform density
            phi = @(q) 1;
            dphi_dt = @(q) 0;
    end
end

function I = polyintegral(fun, vertices)
    % Numerically integrate a function over a polygon using triangular decomposition
    
    if size(vertices,1) < 3
        I = 0;
        return;
    end
    
    % Use the centroid as a reference point
    centroid = mean(vertices, 1);
    
    total_area = 0;
    total_integral = 0;
    
    % Triangulate the polygon (simple fan triangulation)
    for i = 1:size(vertices,1)-1
        % Triangle vertices: centroid, vertex i, vertex i+1
        tri = [centroid; vertices(i,:); vertices(i+1,:)];
        
        % Area of the triangle
        area = 0.5 * abs(det([tri(2,:)-tri(1,:); tri(3,:)-tri(1,:)]));
        total_area = total_area + area;
        
        % Integrate over the triangle using midpoint rule
        midpoint = mean(tri, 1);
        total_integral = total_integral + fun(midpoint') * area;
    end
    
    % Handle last triangle (centroid, last vertex, first vertex)
    tri = [centroid; vertices(end,:); vertices(1,:)];
    area = 0.5 * abs(det([tri(2,:)-tri(1,:); tri(3,:)-tri(1,:)]));
    total_area = total_area + area;
    
    midpoint = mean(tri, 1);
    total_integral = total_integral + fun(midpoint') * area;
    
    I = total_integral;
end

function analyzeConvergence(metrics, C2)
    % Analyze convergence to C2 threshold
    conv_time = zeros(1, size(metrics.centroid_distances,1));
    for i = 1:size(metrics.centroid_distances,1)
        idx = find(metrics.centroid_distances(i,:) < C2, 1);
        if ~isempty(idx)
            conv_time(i) = metrics.time(idx);
        else
            conv_time(i) = inf;
        end
    end
    
    figure('Name', 'Convergence Analysis');
    bar(conv_time);
    title(sprintf('Time to Reach C2=%.2f for Each Agent', C2));
    xlabel('Agent ID'); ylabel('Time (s)');
    grid on;
end

function plotVoronoiEvolution(metrics, selected_agent)
    % Plot evolution of a specific agent's Voronoi cell
    figure('Name', sprintf('Voronoi Cell Evolution - Agent %d', selected_agent));
    
    subplot(2,1,1);
    plot(metrics.time, metrics.voronoi_areas(selected_agent,:));
    title('Cell Area Over Time');
    xlabel('Time (s)'); ylabel('Area');
    grid on;
    
    subplot(2,1,2);
    plot(metrics.time, metrics.voronoi_masses(selected_agent,:));
    title('Cell Mass Over Time');
    xlabel('Time (s)'); ylabel('Mass');
    grid on;
end