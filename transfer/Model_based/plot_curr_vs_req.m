% plot_curr_vs_req.m
% -------------------------------------------------------------
%  Four separate figures comparing actual vs. commanded angles
%  for the actuated joints saved by amber_utils_policy.py.
% -------------------------------------------------------------
function plot_curr_vs_req(csvFile)

    if nargin == 0
        csvFile = 'run_log.csv';   % default file name
    end

    % ---------- read the CSV ----------
    T = readtable(csvFile);

    t = T.sim_time;   % time vector

    % ---------- actual angles ----------
    q1_left_act   = T.q1_left;
    q2_left_act   = T.q2_left;
    q1_right_act  = T.q1_right;
    q2_right_act  = T.q2_right;

    % ---------- reference angles ----------
    q1_left_ref   = T.q1_left_ref;
    q2_left_ref   = T.q2_left_ref;
    q1_right_ref  = T.q1_right_ref;
    q2_right_ref  = T.q2_right_ref;

    % ---------- generate four figures ----------
    makePlot(t, q1_left_act,  q1_left_ref,  'q1\_left');
    makePlot(t, q2_left_act,  q2_left_ref,  'q2\_left');
    makePlot(t, q1_right_act, q1_right_ref, 'q1\_right');
    makePlot(t, q2_right_act, q2_right_ref, 'q2\_right');

end

% -------------------------------------------------------------
% Helper nested function for consistent plot formatting
% -------------------------------------------------------------
function makePlot(t, act, ref, name)
    figure;
    plot(t, act, 'LineWidth', 1.5); hold on;
    plot(t, ref, '--', 'LineWidth', 1.5); hold off;
    xlabel('Time (s)','FontSize',20);
    ylabel('Angle (rad)','FontSize',20);
    set(gca, 'FontSize', 18);  % or any size you prefer
    title([name ' : actual vs. commanded']);
    legend('actual','commanded','Location','best');
    grid on;
end
