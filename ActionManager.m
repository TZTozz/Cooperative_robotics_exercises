classdef ActionManager < handle
    %ACTIONMANAGER Store actions and solve the active task-priority stack.
    %
    % An action is represented by a cell array of Task objects. The order in the cell array defines strict priority: element 1 has the highest priority, and each following task uses only the motion left available by the tasks above it.
    %
    % The solver uses the UVMS system velocity convention
    %
    %   ydot = [q_dot; v_nu]
    %
    % where q_dot contains the arm joint velocities and v_nu contains the vehicle velocities expressed in the vehicle frame. Their dimensions are obtained from the robot model when the manager is constructed.

    properties (Access = private)
        actions          % Dictionary: action name -> prioritized task stack.
        currentActionName (1,1) string = "" % Name of the active action.
        dofArm           % Number of manipulator degrees of freedom.
        dofVehicle       % Number of vehicle velocity degrees of freedom.
        dof              % Total dimension of ydot = [q_dot; v_nu].
        satMin           % [dof x 1] original lower system-velocity limits.
        satMax           % [dof x 1] original upper system-velocity limits.
        ActivationTable dictionary = dictionary(string.empty, cell.empty)
        ActivationFunction
        changingAction = false
        deltaTime = 2    % time to pass from one action to another
    end

    properties (Constant, Access = private)
        % The residual damping objective is generated internally rather than represented by a Task, so its regularization parameters live here.
        residualLambda = 1e-4
        residualThreshold = 0.01
        residualGain = 10
    end

    methods
        function obj = ActionManager(robot)
            %ACTIONMANAGER Construct an empty action library for a UVMS model.
            %
            % Input:
            %   robot - UvmsModel whose arm and vehicle dimensions and velocity limits define the system velocity vector handled by this manager.
            %
            % Output:
            %   obj - Initialized ActionManager with no active action.
            obj.actions = configureDictionary("string", "cell");
            obj.dofArm = numel(robot.q);
            obj.dofVehicle = numel(robot.v_nu);
            obj.dof = obj.dofArm + obj.dofVehicle;

            armMax = robot.q_dot_max(:);
            vehicleMax = robot.v_nu_max(:);
            if numel(armMax) ~= obj.dofArm || numel(vehicleMax) ~= obj.dofVehicle || any(~isfinite([armMax; vehicleMax])) || any([armMax; vehicleMax] < 0)
                error('ActionManager:InvalidVelocityLimits', 'q_dot_max and v_nu_max must contain one finite, non-negative limit for each arm and vehicle velocity.');
            end
            obj.satMax = [armMax; vehicleMax];
            obj.satMin = -obj.satMax;
        end

        function addAction(obj, actionName, taskStack)
            %ADDACTION Register a prioritized task stack under a unique name.
            %
            % Inputs:
            %   actionName - Non-empty string scalar used to identify the action in subsequent calls.
            %   taskStack - 1-by-N cell array of Task objects, ordered from highest to lowest priority.
            %
            % Output:
            %   None. This method updates obj.actions in place.
            actionName = string(actionName);

            if ~isscalar(actionName) || ismissing(actionName) || strlength(strtrim(actionName)) == 0
                error('ActionManager:InvalidActionName', 'Action name must be a non-empty string scalar.');
            end
            actionName = strtrim(actionName);

            if ~iscell(taskStack) || isempty(taskStack) || ~all(cellfun(@(task) isa(task, 'Task'), taskStack))
                error('ActionManager:InvalidTaskStack', 'Task stack must be a non-empty cell array of Task objects.');
            end

            if isKey(obj.actions, actionName)
                error('ActionManager:DuplicateAction', 'An action named "%s" is already registered.', actionName);
            end

            % Dictionary values have cell type; the outer cell stores the complete task stack as the single value associated with the key.
            obj.actions(actionName) = {taskStack};
        end

        function [v_nu, q_dot] = computeICAT(obj, robot)
            %COMPUTEICAT Compute arm and vehicle velocity references.
            %
            % Input:
            %   robot - UvmsModel containing the current state, transforms, goals and sensor data used by the active tasks.
            %
            % Outputs:
            %   v_nu - [dofVehicle x 1] vehicle velocity expressed in the vehicle frame.
            %   q_dot - [dofArm x 1] manipulator joint velocity [rad/s].
            %
            % Each task first updates its reference rate xdotbar, Jacobian J and activation matrix A. The iCAT recursion then processes the tasks in priority order while preserving higher-priority motion.

            if obj.currentActionName == ""
                error('ActionManager:NoActiveAction', 'Select an action before calling computeICAT.');
            end

            % Select the prioritized list associated with the active action.
            if (obj.changingAction)         % In case the action is changing
                tasks = obj.actions{'unified_list'};
    
                % Refresh all configuration-dependent task quantities before the stack is inverted. A task may use robot state in all three calls.
                for i = 1:length(tasks)
                    tasks{i}.updateReference(robot);
                    tasks{i}.updateJacobian(robot);
                    tasks{i}.updateActivation(robot);
                    tasks{i}.A = tasks{i}.A * obj.ActivationFunction(i);
                end
            else
                tasks = obj.actions{obj.currentActionName};
    
                % Refresh all configuration-dependent task quantities before the stack is inverted. A task may use robot state in all three calls.
                for i = 1:length(tasks)
                    tasks{i}.updateReference(robot);
                    tasks{i}.updateJacobian(robot);
                    tasks{i}.updateActivation(robot);
                end
            end

            % Start with zero commanded velocity, the full system motion space and the original velocity bounds. Each priority consumes part of the available velocity range.
            ydotbar = zeros(obj.dof,1);
            Qp = eye(obj.dof);
            satMinRemaining = obj.satMin;            
            satMaxRemaining = obj.satMax;
            K = eye(obj.dof); % Neutral control-space weighting required by the current iCAT interface.
            for i = 1:length(tasks)
                [Qp, ydotbar, ~, ~, ~, satMinRemaining, satMaxRemaining] = iCAT_task(tasks{i}.A, tasks{i}.J, Qp, ydotbar, tasks{i}.xdotbar, tasks{i}.lambda, tasks{i}.threshold, tasks{i}.gain, satMinRemaining, satMaxRemaining, obj.satMin, obj.satMax, K);
            end

            % Use any residual degrees of freedom to drive the system velocity toward zero. As the last-priority task, this cannot alter motion already assigned to higher-priority objectives and must respect the velocity range left by the preceding tasks.
            [~, ydotbar] = iCAT_task(eye(obj.dof), eye(obj.dof), Qp, ydotbar, zeros(obj.dof,1), obj.residualLambda, obj.residualThreshold, obj.residualGain, satMinRemaining, satMaxRemaining, obj.satMin, obj.satMax, K);

            % Split ydotbar using the same ordering used to build every task Jacobian: arm joints first, then the vehicle twist.
            q_dot = ydotbar(1:obj.dofArm);
            v_nu = ydotbar(obj.dofArm+1:obj.dof);
        end

        function setCurrentAction(obj, actionName)
            %SETCURRENTACTION Select an action by name.
            %
            % Input:
            %   actionName - String scalar previously passed to addAction.
            %
            % Output:
            %   None. This method updates obj.currentActionName in place.
            actionName = string(actionName);

            if ~isscalar(actionName) || ismissing(actionName) || strlength(strtrim(actionName)) == 0
                error('ActionManager:InvalidActionName', 'Action name must be a non-empty string scalar.');
            end
            actionName = strtrim(actionName);

            if ~isKey(obj.actions, actionName)
                error('ActionManager:UnknownAction', 'No action named "%s" has been registered.', actionName);
            end

            obj.currentActionName = actionName;
        end


        function compileSmoothChange(obj)
            unified = obj.actions{"unified_list"};
            unified_names = cellfun(@(x) string(class(x)), unified);
            
            action_keys = obj.actions.keys;
            action_keys = action_keys(action_keys ~= "unified_list");
            
            for i = 1:length(action_keys)
                k = action_keys(i);
                action_list = obj.actions{k};
                
                % All zeros
                mask = zeros(size(unified));
        
                if ~isempty(action_list)
                    action_names = cellfun(@(x) string(class(x)), action_list); % Extract the name of the task
                    
                    j = 1; % Pointer to the task that must be found inside the unified list
                    for idx = 1:length(unified_names) % Pointer of the unified list
                        if j <= length(action_names) && unified_names(idx) == action_names(j)
                            mask(idx) = 1;
                            j = j + 1;
                        end
                    end
                    
                    % Check if all tasks were found
                    if j <= length(action_names)
                        error('The action "%s" is not inside the unified list.', k);
                    end
                end
                
                obj.ActivationTable{k} = mask;
            end
            
            disp(obj.ActivationTable);
        end


        function [Done] = SmoothActivation(obj, time, previous, new)
            % Create the vector that rapresent the changing from one input to the other
            % Input:
            %     time: time passed
            %     previous: name of the prevoius task
            %     new: name of the new task
            %
            % Output:
            %     Done: boolean value that indicates that the changing phase ended
            %

            if (time >= obj.deltaTime)      % in case the changing phase ended
                obj.setCurrentAction(new);
                obj.changingAction = false;
                Done = true;
                return
            end

            if(obj.changingAction == false)
                obj.changingAction = true;
            end

            vec = obj.ActivationTable{previous} - 2 * obj.ActivationTable{new};

            for i = 1:length(vec)   % For each task in the unified list
                switch vec(i)
                    case 0          % 00
                        obj.ActivationFunction(i) = 0;
                    case -2         % 01
                        obj.ActivationFunction(i) = IncreasingBellShapedFunction(0, obj.deltaTime, 0, 1, time);
                    case 1          % 10
                        obj.ActivationFunction(i) = DecreasingBellShapedFunction(0, obj.deltaTime, 0, 1, time);
                    case -1         % 11
                        obj.ActivationFunction(i) = 1;
                end
            end

            Done = false;

        end
    end
end
