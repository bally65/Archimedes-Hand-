close % 关闭当前的Figure窗口
clear % 清除工作空间的所有变量
clc   % 清除命令窗口的内容，对工作环境中的全部变量无任何影响

% 添加模型路径
allPaths = genpath('URDF_XG_Robot_Arm_Urdf_V1_1');
addpath(allPaths);

% 导入urdf模型
robot = importrobot('URDF_XG_Robot_Arm_Urdf_V1_1.urdf');
% 打开一个窗口显示机器人
showdetails(robot)
ax = show(robot, 'Frames','on','Visuals','on');

% 模型导入simulink
% robot_sm = smimport('URDF_XG_Robot_Arm_Urdf_V1_1.urdf');
