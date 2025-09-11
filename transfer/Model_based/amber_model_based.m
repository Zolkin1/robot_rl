% Replace 'my_run.csv' with your actual path/filename:
csvFile = 'amber_joint_log.csv';

% Read entire table (assumes header row exactly matches what we wrote)
T = readtable(csvFile);

% Time vector
time = T.sim_time;

% ———————————————————————————————
% 1) Actual vs. commanded joint angles
% ———————————————————————————————
q1_left_act   = T.q1_left;
q2_left_act   = T.q2_left;
q1_right_act  = T.q1_right;
q2_right_act  = T.q2_right;

q1_left_req   = T.act_q1_left;
q2_left_req   = T.act_q2_left;
q1_right_req  = T.act_q1_right;
q2_right_req  = T.act_q2_right;

% Plot q1_left
figure;
plot(time, q1_left_act, 'LineWidth',1.5); hold on;
plot(time, q1_left_req, '--', 'LineWidth',1.5); hold off;
xlabel('Time (s)'); ylabel('Angle (rad)');
title('q1\_left: Actual vs. Commanded');
legend('actual','command','Location','best');
grid on;

% Plot q2_left
figure;
plot(time, q2_left_act, 'LineWidth',1.5); hold on;
plot(time, q2_left_req, '--', 'LineWidth',1.5); hold off;
xlabel('Time (s)'); ylabel('Angle (rad)');
title('q2\_left: Actual vs. Commanded');
legend('actual','command','Location','best');
grid on;

% Plot q1_right
figure;
plot(time, q1_right_act, 'LineWidth',1.5); hold on;
plot(time, q1_right_req, '--', 'LineWidth',1.5); hold off;
xlabel('Time (s)'); ylabel('Angle (rad)');
title('q1\_right: Actual vs. Commanded');
legend('actual','command','Location','best');
grid on;

% Plot q2_right
figure;
plot(time, q2_right_act, 'LineWidth',1.5); hold on;
plot(time, q2_right_req, '--', 'LineWidth',1.5); hold off;
xlabel('Time (s)'); ylabel('Angle (rad)');
title('q2\_right: Actual vs. Commanded');
legend('actual','command','Location','best');
grid on;


% ———————————————————————————————
% 2) COM Trajectory (x,y,z)
% ———————————————————————————————
com_x = T.com_x;
com_y = T.com_y;
com_z = T.com_z;

figure;
plot(time, com_x, 'LineWidth',1.5); hold on;
plot(time, com_y, 'LineWidth',1.5);
plot(time, com_z, 'LineWidth',1.5); hold off;
set(gca, 'FontSize', 18);  % or any size you prefer
xlabel('Time (s)','FontSize', 18);
ylabel('CoM Position (m)','FontSize', 18);
title('Center of Mass Trajectory');
legend('COM\_x','COM\_y','COM\_z','Location','best');
grid on;

