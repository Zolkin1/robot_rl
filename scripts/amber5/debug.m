function plot_amber_csv(csvFile)
% PLOT_AMBER_CSV  Plot joint positions from the Amber joint-log CSV.
%
%   PLOT_AMBER_CSV           – looks for "amber_joint_log.csv" in cwd
%   PLOT_AMBER_CSV(filename) – use a specific CSV file
%
% The script creates eight figures, one for each joint:
%   PrismaticJoint, base_link_to_base_link2, base_link2_to_base_link3,
%   base_link3_to_torso, q1_left, q1_right, q2_left, q2_right
%
% Each figure shows one curve per environment (env_id).

    if nargin == 0
        csvFile = "logs/datalog.csv";
    end

    % ---------- read the CSV ----------
    fprintf('[INFO] Loading %s …\n', csvFile);
    T = readtable(csvFile, "PreserveVariableNames", true);

    % ---------- check the requested joints exist ----------
    % joints = { ...
    %     "PrismaticJoint", ...
    %     "base_link_to_base_link2", ...
    %     "base_link2_to_base_link3", ...
    %     "base_link3_to_torso", ...
    %     "q1_left", "q1_right", "q2_left", "q2_right" ...
    % };
    joints = [ ...
    "PrismaticJoint", ...
    "base_link_to_base_link2", ...
    "base_link2_to_base_link3", ...
    "base_link3_to_torso", ...
    "q1_left", "q1_right", "q2_left", "q2_right" ...
];
    missing = setdiff(joints, T.Properties.VariableNames);
    if ~isempty(missing)
        warning('The following joints were NOT found in the CSV and will be skipped:\n  %s', strjoin(missing, ", "));
        joints = setdiff(joints, missing);    % drop them
    end

    % ---------- loop over joints ----------
    envs = unique(T.env_id);          % vector of environment IDs
    colors = lines(numel(envs));      % distinct colours for each env

    for j = 1:numel(joints)
        jointName = joints{j};

        figure('Name', jointName);  hold on;  grid on;
        for k = 1:numel(envs)
            envMask = T.env_id == envs(k);
            plot(T.sim_time(envMask), T.(jointName)(envMask), ...
                'DisplayName', sprintf('env %d', envs(k)), ...
                'Color', colors(k, :));
        end
        xlabel('sim\_time  [s]');
        ylabel(sprintf('%s  [rad or m]', jointName), 'Interpreter', 'none');
        title(jointName, 'Interpreter', 'none');
        legend('Location', 'best');
    end
end
