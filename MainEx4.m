%% ROBUST UVMS TASK-PRIORITY EXERCISE
% This script assembles the main blocks of the control architecture:
%
%   task definitions -> ActionManager/iCAT -> kinematic simulator -> Unity
%
% At every iteration, the Kinematic Control Layer computes the system velocity ydot = [q_dot; v_nu].
% UvmsSim integrates this reference directly; in a real robot, a Dynamic Control Layer would track the same reference.
%
% Run this script with the current MATLAB folder set to the template root so that the relative paths below resolve consistently.

%% Make exercise dependencies visible to MATLAB
addpath('./tools');
addpath('./icat');
addpath('./robust_robot');
addpath('./tpik');
clc; clear; close all;

%% Configure and initialize the simulation
% dt is both the integration step and the nominal controller sample time.
dt       = 0.005;
endTime  = 50;

% UvmsModel owns state and kinematics; UvmsSim updates that shared handle.
robotModel = UvmsModel();
sim = UvmsSim(dt, robotModel, endTime);

% Initial position
robotModel.eta = [48.5 11.5 -33 -pi/3 0 pi/2]';%[10.5 35.5 -36 -pi/3 pi/3 pi/2]';[48.5 11.5 -33] 

% Unity is used only as viewer/sensor endpoint; it is not the simulator.
unity = UnityInterface("127.0.0.1");

%% Define one action as a strict-priority stack of tasks
% The first cell is the highest-priority task. Add safety tasks before the action objective when extending this exercise (for example, joint limits).
task_tool = TaskTool();
task_position = TaskPosition();
task_h_attitude = TaskHorizontalAttitude();
task_min_altitude = TaskAltitude();
task_landing = TaskLanding();

move_vehicle_tasks = {task_min_altitude task_h_attitude task_position};
landing_vehicle_task = {task_landing task_h_attitude task_position};

unified_task_vector = {task_landing task_min_altitude task_h_attitude task_position};

% Actions are registered and selected by name; the task order still defines strict priority within each action.
actionManager = ActionManager(robotModel);
actionManager.addAction("safe_navigation", move_vehicle_tasks);
actionManager.addAction("landing", landing_vehicle_task);
actionManager.addAction("unified_list", unified_task_vector)
actionManager.setCurrentAction('safe_navigation');

% The logger monitors the tasks belonging to the action selected above.
task_set = move_vehicle_tasks;

actionManager.compileSmoothChange;
changingAction = false;     % Inizialize flag
Done = false;

%% Set desired poses
% Positions are in metres and orientations are [roll pitch yaw] in radians. Both poses are expressed in the inertial/world frame.
w_arm_goal_position = [12.2025, 37.3748, -39.8860]';
w_arm_goal_orientation = [0, pi, pi/2];
w_vehicle_goal_position = [50 -12.5 -33]';
w_vehicle_goal_orientation = [0, 0, 0];

% Store the goals as homogeneous transforms in the shared model.
robotModel.setGoal(w_arm_goal_position, w_arm_goal_orientation, w_vehicle_goal_position, w_vehicle_goal_orientation);

% Preallocate one column for the initial state plus all integration steps.
logger = SimulationLogger(ceil(endTime/dt)+1, robotModel, task_set);

%% Run the closed-loop kinematic simulation
tic
for step = 1:sim.maxSteps
    % Read the newest sensor sample. If none is available, the interface keeps the previous altitude value.
    robotModel.altitude = unity.receiveAltitude(robotModel);

    % Start the smooth changing between actions
    if ((norm(robotModel.wTv(1:3,4)-robotModel.vehicleGoalPosition) < 0.1) && ~Done && ~changingAction)
        disp("Started the smooth changing");
        changingAction = true;
        startTime = sim.time;
    end

    if changingAction
        Done = actionManager.SmoothActivation(sim.time - startTime, 'safe_navigation', 'landing');
        if Done
            changingAction = false;
            disp("Smooth changing done");
        end
    end

    % Update every task and solve the current strict-priority hierarchy.
    [v_nu, q_dot] = actionManager.computeICAT(robotModel);

    % Integrate arm and vehicle velocities, then refresh all transforms.
    sim.step(v_nu, q_dot);

    % Send the state after integration so the viewer displays the new sample.
    unity.send(robotModel);

    % Store state, task references and activations at the current sample.
    logger.update(sim.time, sim.loopCounter);

    % Print a compact status update approximately once per simulated second.
    if mod(sim.loopCounter, round(0.5 / sim.dt)) == 0
        fprintf('t = %.2f s\n', sim.time);
        fprintf('alt = %.2f m\n', robotModel.altitude);
        fprintf('dist = %.2f m\n', (robotModel.wTv(1:3,4)-robotModel.vehicleGoalPosition));
        % fprintf('attitude = %.2f rad\n', robotModel.eta(4));
        fprintf('A: %.2f\n', task_landing.A);
    end

    % Busy-wait until one sample period has elapsed when real-time playback is desired. 
    % Remove this call to execute the simulation as fast as possible.
    %SlowdownToRealtime(dt);
end

%% Review results and release communication resources
logger.plotAll();

% Explicit deletion closes UDP sockets before the script terminates.
delete(unity);
