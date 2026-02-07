close % 关闭当前的Figure窗口
clear % 清除工作空间的所有变量
clc   % 清除命令窗口的内容，对工作环境中的全部变量无任何影响

% 添加模型路径
allPaths = genpath('.');
addpath(allPaths);

% 初始化全局变量
D_H = [
    0,    0,   0,   pi/2;
    0,    pi/2,    0,   pi/2;
    200,  pi,    0,  -pi/2;
    47.63, -pi/2, -184.5,  0;
    0,    pi/2,    0,   pi/2;
    0,    pi/2,  0,    0
]; 

joints_theta = [0,  0,  0, 0, 0, -180]; % 关节角度deg

% 打印各关节变化矩阵
robot_analyse(D_H)

% 目标姿态矩阵
% T_target = [0 -1 0 0;
%             0 0 -1 -47.63;
%             1 0 0 15.5;
%             0 0 0 1];

T_target = [0 -1 0 0;
            0 0 -1 -200;
            1 0 0 15.5;
            0 0 0 1];


% 打开simulink
open_system('URDF_XG_Robot_Arm_Urdf_Control_V3.slx');
