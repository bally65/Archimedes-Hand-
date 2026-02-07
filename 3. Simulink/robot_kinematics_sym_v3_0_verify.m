clc
clear
% 机械臂运动学逆解析，根据D-H表，推理出各关节角度符号公式, 使用pieper的解法, 验证算法
D_H = [
    0,    0,   0,   pi/2;
    0,    pi/2,    0,   pi/2;
    200,  pi,    0,  -pi/2;
    47.63, -pi/2, -184.5,  0;
    0,    pi/2,    0,   pi/2;
    0,    pi/2,  0,    0
];

T_target = [0 -1 0 0;
            0 0 -1 -47.63;
            1 0 0 15.5;
            0 0 0 1];

% T_target = [0 -1 0 0;
%             0 0 -1 -100;
%             1 0 0 15.5;
%             0 0 0 1];

a_cow = D_H(:, 1);
alpha_cow = D_H(:, 2);
d_cow = D_H(:, 3);

a2 = a_cow(3);
a3 = a_cow(4);
d4 = d_cow(4);
px = T_target(1,4);
py = T_target(2,4);
pz = T_target(3,4);

% theta3求解
u1_theta3 = -(2*a2*d4 + (- a2^4 + 2*a2^2*a3^2 + 2*a2^2*d4^2 + 2*a2^2*px^2 + 2*a2^2*py^2 + 2*a2^2*pz^2 - a3^4 - 2*a3^2*d4^2 + 2*a3^2*px^2 + 2*a3^2*py^2 + 2*a3^2*pz^2 - d4^4 + 2*d4^2*px^2 + 2*d4^2*py^2 + 2*d4^2*pz^2 - px^4 - 2*px^2*py^2 - 2*px^2*pz^2 - py^4 - 2*py^2*pz^2 - pz^4)^(1/2))/(- a2^2 + 2*a2*a3 - a3^2 - d4^2 + px^2 + py^2 + pz^2);
u2_theta3 = -(2*a2*d4 - (- a2^4 + 2*a2^2*a3^2 + 2*a2^2*d4^2 + 2*a2^2*px^2 + 2*a2^2*py^2 + 2*a2^2*pz^2 - a3^4 - 2*a3^2*d4^2 + 2*a3^2*px^2 + 2*a3^2*py^2 + 2*a3^2*pz^2 - d4^4 + 2*d4^2*px^2 + 2*d4^2*py^2 + 2*d4^2*pz^2 - px^4 - 2*px^2*py^2 - 2*px^2*pz^2 - py^4 - 2*py^2*pz^2 - pz^4)^(1/2))/(- a2^2 + 2*a2*a3 - a3^2 - d4^2 + px^2 + py^2 + pz^2);
theta3_1 = atan(u1_theta3)*2;
theta3_2 = atan(u2_theta3)*2;
disp("theta3_1 = ");
disp(rad2deg(theta3_1));
disp("theta3_2 = ");
disp(rad2deg(theta3_2));
theta3_value_test = theta3_1;

% theta2求解
theta3 = theta3_value_test;
u1_theta2 = -(a2 + (a2^2 + 2*cos(theta3)*a2*a3 - 2*sin(theta3)*a2*d4 + a3^2 + d4^2 - pz^2)^(1/2) + a3*cos(theta3) - d4*sin(theta3))/(d4*cos(theta3) - pz + a3*sin(theta3));
u2_theta2 = -(a2 - (a2^2 + 2*cos(theta3)*a2*a3 - 2*sin(theta3)*a2*d4 + a3^2 + d4^2 - pz^2)^(1/2) + a3*cos(theta3) - d4*sin(theta3))/(d4*cos(theta3) - pz + a3*sin(theta3));
theta2_1 = atan(u1_theta2)*2;
theta2_2 = atan(u2_theta2)*2;
disp("theta2_1 = ");
disp(rad2deg(theta2_1));
disp("theta2_2 = ");
disp(rad2deg(theta2_2));
theta2_value_test = theta2_1;

% theta1求解
theta2 = theta2_value_test;
u1_theta1 =  ((a2*cos(theta2) - px + a3*cos(theta2 - theta3) + d4*sin(theta2 - theta3))/(px + a2*cos(theta2) + a3*cos(theta2 - theta3) + d4*sin(theta2 - theta3)))^(1/2);
u1_theta2 = -((a2*cos(theta2) - px + a3*cos(theta2 - theta3) + d4*sin(theta2 - theta3))/(px + a2*cos(theta2) + a3*cos(theta2 - theta3) + d4*sin(theta2 - theta3)))^(1/2);
theta1_1 = atan(u1_theta1)*2;
theta1_2 = atan(u1_theta2)*2;
disp("theta1_1 = ");
disp(rad2deg(theta1_1));
disp("theta1_2 = ");
disp(rad2deg(theta1_2));
theta1_value_test = theta1_1;


% 欧拉角z-y-z
% theta5求解
theta4 = 0; % 欧拉角求解时, theta4=0, 表示初始状态
theta1 = theta1_value_test;
nx = T_target(1,1);
ny = T_target(2,1);
nz = T_target(3,1);
ox = T_target(1,2);
oy = T_target(2,2);
oz = T_target(3,2);
ax= T_target(1,3);
ay= T_target(2,3);
az= T_target(3,3);
r32 = - (oz*(cos(theta2)*cos(theta3) + sin(theta2)*sin(theta3)))/(cos(theta2)^2*cos(theta3)^2 + cos(theta2)^2*sin(theta3)^2 + cos(theta3)^2*sin(theta2)^2 + sin(theta2)^2*sin(theta3)^2) - (ox*(cos(theta1)*cos(theta2)*sin(theta3) - cos(theta1)*cos(theta3)*sin(theta2)))/(cos(theta1)^2*cos(theta2)^2*cos(theta3)^2 + cos(theta1)^2*cos(theta2)^2*sin(theta3)^2 + cos(theta1)^2*cos(theta3)^2*sin(theta2)^2 + cos(theta1)^2*sin(theta2)^2*sin(theta3)^2 + cos(theta2)^2*cos(theta3)^2*sin(theta1)^2 + cos(theta2)^2*sin(theta1)^2*sin(theta3)^2 + cos(theta3)^2*sin(theta1)^2*sin(theta2)^2 + sin(theta1)^2*sin(theta2)^2*sin(theta3)^2) - (oy*(cos(theta2)*sin(theta1)*sin(theta3) - cos(theta3)*sin(theta1)*sin(theta2)))/(cos(theta1)^2*cos(theta2)^2*cos(theta3)^2 + cos(theta1)^2*cos(theta2)^2*sin(theta3)^2 + cos(theta1)^2*cos(theta3)^2*sin(theta2)^2 + cos(theta1)^2*sin(theta2)^2*sin(theta3)^2 + cos(theta2)^2*cos(theta3)^2*sin(theta1)^2 + cos(theta2)^2*sin(theta1)^2*sin(theta3)^2 + cos(theta3)^2*sin(theta1)^2*sin(theta2)^2 + sin(theta1)^2*sin(theta2)^2*sin(theta3)^2);
r31 = - (nz*(cos(theta2)*cos(theta3) + sin(theta2)*sin(theta3)))/(cos(theta2)^2*cos(theta3)^2 + cos(theta2)^2*sin(theta3)^2 + cos(theta3)^2*sin(theta2)^2 + sin(theta2)^2*sin(theta3)^2) - (nx*(cos(theta1)*cos(theta2)*sin(theta3) - cos(theta1)*cos(theta3)*sin(theta2)))/(cos(theta1)^2*cos(theta2)^2*cos(theta3)^2 + cos(theta1)^2*cos(theta2)^2*sin(theta3)^2 + cos(theta1)^2*cos(theta3)^2*sin(theta2)^2 + cos(theta1)^2*sin(theta2)^2*sin(theta3)^2 + cos(theta2)^2*cos(theta3)^2*sin(theta1)^2 + cos(theta2)^2*sin(theta1)^2*sin(theta3)^2 + cos(theta3)^2*sin(theta1)^2*sin(theta2)^2 + sin(theta1)^2*sin(theta2)^2*sin(theta3)^2) - (ny*(cos(theta2)*sin(theta1)*sin(theta3) - cos(theta3)*sin(theta1)*sin(theta2)))/(cos(theta1)^2*cos(theta2)^2*cos(theta3)^2 + cos(theta1)^2*cos(theta2)^2*sin(theta3)^2 + cos(theta1)^2*cos(theta3)^2*sin(theta2)^2 + cos(theta1)^2*sin(theta2)^2*sin(theta3)^2 + cos(theta2)^2*cos(theta3)^2*sin(theta1)^2 + cos(theta2)^2*sin(theta1)^2*sin(theta3)^2 + cos(theta3)^2*sin(theta1)^2*sin(theta2)^2 + sin(theta1)^2*sin(theta2)^2*sin(theta3)^2);
r33 = - (az*(cos(theta2)*cos(theta3) + sin(theta2)*sin(theta3)))/(cos(theta2)^2*cos(theta3)^2 + cos(theta2)^2*sin(theta3)^2 + cos(theta3)^2*sin(theta2)^2 + sin(theta2)^2*sin(theta3)^2) - (ax*(cos(theta1)*cos(theta2)*sin(theta3) - cos(theta1)*cos(theta3)*sin(theta2)))/(cos(theta1)^2*cos(theta2)^2*cos(theta3)^2 + cos(theta1)^2*cos(theta2)^2*sin(theta3)^2 + cos(theta1)^2*cos(theta3)^2*sin(theta2)^2 + cos(theta1)^2*sin(theta2)^2*sin(theta3)^2 + cos(theta2)^2*cos(theta3)^2*sin(theta1)^2 + cos(theta2)^2*sin(theta1)^2*sin(theta3)^2 + cos(theta3)^2*sin(theta1)^2*sin(theta2)^2 + sin(theta1)^2*sin(theta2)^2*sin(theta3)^2) - (ay*(cos(theta2)*sin(theta1)*sin(theta3) - cos(theta3)*sin(theta1)*sin(theta2)))/(cos(theta1)^2*cos(theta2)^2*cos(theta3)^2 + cos(theta1)^2*cos(theta2)^2*sin(theta3)^2 + cos(theta1)^2*cos(theta3)^2*sin(theta2)^2 + cos(theta1)^2*sin(theta2)^2*sin(theta3)^2 + cos(theta2)^2*cos(theta3)^2*sin(theta1)^2 + cos(theta2)^2*sin(theta1)^2*sin(theta3)^2 + cos(theta3)^2*sin(theta1)^2*sin(theta2)^2 + sin(theta1)^2*sin(theta2)^2*sin(theta3)^2);
theta5 = atan2(sqrt(r32^2 + r31^2), r33);
theta5 = -theta5 + pi;
disp("theta5 = ");
disp(rad2deg(theta5));

% theta4求解
disp("if theta5 == 0, theta4 = 0, theta6 = atan2(-r12, r11)")
disp("if theta5 == 180, theta4 = 0, theta6 = atan2(-r12, r11)")
r23 = -(ax*cos(theta1)*cos(theta2)*cos(theta3)*sin(theta4) - ax*cos(theta4)*sin(theta1)*sin(theta2)^2*sin(theta3)^2 + ay*cos(theta2)*cos(theta3)*sin(theta1)*sin(theta4) + ax*cos(theta1)*sin(theta2)*sin(theta3)*sin(theta4) + ay*sin(theta1)*sin(theta2)*sin(theta3)*sin(theta4) - az*cos(theta1)^2*cos(theta2)*sin(theta3)*sin(theta4) + az*cos(theta1)^2*cos(theta3)*sin(theta2)*sin(theta4) - az*cos(theta2)*sin(theta1)^2*sin(theta3)*sin(theta4) + az*cos(theta3)*sin(theta1)^2*sin(theta2)*sin(theta4) + ay*cos(theta1)*cos(theta2)^2*cos(theta3)^2*cos(theta4) - ax*cos(theta2)^2*cos(theta3)^2*cos(theta4)*sin(theta1) + ay*cos(theta1)*cos(theta2)^2*cos(theta4)*sin(theta3)^2 + ay*cos(theta1)*cos(theta3)^2*cos(theta4)*sin(theta2)^2 - ax*cos(theta2)^2*cos(theta4)*sin(theta1)*sin(theta3)^2 - ax*cos(theta3)^2*cos(theta4)*sin(theta1)*sin(theta2)^2 + ay*cos(theta1)*cos(theta4)*sin(theta2)^2*sin(theta3)^2)/((cos(theta1)^2*cos(theta4)^2 + cos(theta1)^2*sin(theta4)^2 + cos(theta4)^2*sin(theta1)^2 + sin(theta1)^2*sin(theta4)^2)*(cos(theta2)^2*cos(theta3)^2 + cos(theta2)^2*sin(theta3)^2 + cos(theta3)^2*sin(theta2)^2 + sin(theta2)^2*sin(theta3)^2));
r13 = (-az*cos(theta4)*cos(theta1)^2*cos(theta2)*sin(theta3) + az*cos(theta4)*cos(theta1)^2*cos(theta3)*sin(theta2) - ay*sin(theta4)*cos(theta1)*cos(theta2)^2*cos(theta3)^2 - ay*sin(theta4)*cos(theta1)*cos(theta2)^2*sin(theta3)^2 + ax*cos(theta4)*cos(theta1)*cos(theta2)*cos(theta3) - ay*sin(theta4)*cos(theta1)*cos(theta3)^2*sin(theta2)^2 - ay*sin(theta4)*cos(theta1)*sin(theta2)^2*sin(theta3)^2 + ax*cos(theta4)*cos(theta1)*sin(theta2)*sin(theta3) + ax*sin(theta4)*cos(theta2)^2*cos(theta3)^2*sin(theta1) + ax*sin(theta4)*cos(theta2)^2*sin(theta1)*sin(theta3)^2 + ay*cos(theta4)*cos(theta2)*cos(theta3)*sin(theta1) - az*cos(theta4)*cos(theta2)*sin(theta1)^2*sin(theta3) + ax*sin(theta4)*cos(theta3)^2*sin(theta1)*sin(theta2)^2 + az*cos(theta4)*cos(theta3)*sin(theta1)^2*sin(theta2) + ax*sin(theta4)*sin(theta1)*sin(theta2)^2*sin(theta3)^2 + ay*cos(theta4)*sin(theta1)*sin(theta2)*sin(theta3))/((cos(theta1)^2 + sin(theta1)^2)*(cos(theta4)^2 + sin(theta4)^2)*(cos(theta2)^2*cos(theta3)^2 + cos(theta2)^2*sin(theta3)^2 + cos(theta3)^2*sin(theta2)^2 + sin(theta2)^2*sin(theta3)^2));
theta4_value = atan2(r23, r13);
disp("theta4 = ");
disp(rad2deg(theta4_value));

% theta6求解
theta6_value = atan2(r32, -r31);
theta6_value = theta6_value - pi;
disp("theta6 = ");
disp(rad2deg(theta6_value));

thetas = [rad2deg(theta1_value_test), rad2deg(theta2_value_test), rad2deg(theta3_value_test), rad2deg(theta4_value), rad2deg(theta5), rad2deg(theta6_value)];
disp("thetas = ")
disp(thetas)

joints_theta = [rad2deg(theta1_value_test - pi/2),  rad2deg(theta2_value_test - pi/2),  rad2deg(theta3_value_test + pi/2), rad2deg(theta4_value - pi), rad2deg(theta5 - pi/2), rad2deg(theta6_value)]; % 关节角度deg
disp("joints_theta = ")
disp(joints_theta)




