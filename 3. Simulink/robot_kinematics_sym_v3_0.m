clc
clear
% 机械臂运动学逆解析，根据D-H表，推理出各关节角度符号公式, 使用pieper的解法
syms a2 a3 d4
D_H = [
    0,    0,   0,   pi/2;
    0,    pi/2,    0,   pi/2;
    a2,  pi,    0,  -pi/2;
    a3, -pi/2, d4,  0;
    0,    pi/2,    0,   pi/2;
    0,    pi/2,  0,    0
];
disp("D_H:")
disp(D_H)

syms nx ny nz ox oy oz ax ay az px py pz
T_target = [nx ox ax px;
            ny oy ay py;
            nz oz az pz;
            0 0 0 1];

a_cow = D_H(:, 1);
alpha_cow = D_H(:, 2);
d_cow = D_H(:, 3);

% T0_1
syms theta1
theta = theta1;
D_H_row = D_H(1,:);
a = D_H_row(1);
alpha = D_H_row(2)/pi;
d = D_H_row(3);

T0_1 = [cos(theta), -sin(theta), 0, a;
    sin(theta)*cospi(alpha), cos(theta)*cospi(alpha), -sinpi(alpha), -d*sinpi(alpha);
    sin(theta)*sinpi(alpha), cos(theta)*sinpi(alpha), cospi(alpha), d*cospi(alpha);
    0, 0, 0, 1];
disp("T0_1:")
disp(T0_1)

% T1_2
syms theta2
theta = theta2;
D_H_row = D_H(2,:);
a = D_H_row(1);
alpha = D_H_row(2)/pi;
d = D_H_row(3);

T1_2 = [cos(theta), -sin(theta), 0, a;
    sin(theta)*cospi(alpha), cos(theta)*cospi(alpha), -sinpi(alpha), -d*sinpi(alpha);
    sin(theta)*sinpi(alpha), cos(theta)*sinpi(alpha), cospi(alpha), d*cospi(alpha);
    0, 0, 0, 1];
disp("T1_2:")
disp(T1_2)

% T2_3
syms theta3
theta = theta3;
D_H_row = D_H(3,:);
a = D_H_row(1);
alpha = D_H_row(2)/pi;
d = D_H_row(3);

T2_3 = [cos(theta), -sin(theta), 0, a;
    sin(theta)*cospi(alpha), cos(theta)*cospi(alpha), -sinpi(alpha), -d*sinpi(alpha);
    sin(theta)*sinpi(alpha), cos(theta)*sinpi(alpha), cospi(alpha), d*cospi(alpha);
    0, 0, 0, 1];
disp("T2_3:")
disp(T2_3)

% T3_4
syms theta4
theta = theta4;
D_H_row = D_H(4,:);
a = D_H_row(1);
alpha = D_H_row(2)/pi;
d = D_H_row(3);

T3_4 = [cos(theta), -sin(theta), 0, a;
    sin(theta)*cospi(alpha), cos(theta)*cospi(alpha), -sinpi(alpha), -d*sinpi(alpha);
    sin(theta)*sinpi(alpha), cos(theta)*sinpi(alpha), cospi(alpha), d*cospi(alpha);
    0, 0, 0, 1];
disp("T3_4:")
disp(T3_4)

% T4_5
syms theta5
theta = theta5;
D_H_row = D_H(5,:);
a = D_H_row(1);
alpha = D_H_row(2)/pi;
d = D_H_row(3);

T4_5 = [cos(theta), -sin(theta), 0, a;
    sin(theta)*cospi(alpha), cos(theta)*cospi(alpha), -sinpi(alpha), -d*sinpi(alpha);
    sin(theta)*sinpi(alpha), cos(theta)*sinpi(alpha), cospi(alpha), d*cospi(alpha);
    0, 0, 0, 1];
disp("T4_5:")
disp(T4_5)

% T5_6
syms theta6
theta = theta6;
D_H_row = D_H(6,:);
a = D_H_row(1);
alpha = D_H_row(2)/pi;
d = D_H_row(3);

T5_6 = [cos(theta), -sin(theta), 0, a;
    sin(theta)*cospi(alpha), cos(theta)*cospi(alpha), -sinpi(alpha), -d*sinpi(alpha);
    sin(theta)*sinpi(alpha), cos(theta)*sinpi(alpha), cospi(alpha), d*cospi(alpha);
    0, 0, 0, 1];
disp("T5_6:")
disp(T5_6)
   
syms f1 f2 f3 g1 g2 g3
f = T2_3 * T3_4(:,4);
g = T1_2 * [f1;f2;f3;1];
P0_4_ORG = T0_1 * [g1;g2;g3;1];

r = simplify(P0_4_ORG(1)^2 + P0_4_ORG(2)^2 + P0_4_ORG(3)^2);
r = simplify(subs(r, [g1 g2 g3], [g(1) g(2) g(3)]));
r = simplify(subs(r, [f1 f2 f3], [f(1) f(2) f(3)]));

% theta3求解
fprintf("\n\n****************[Solving theta3]****************\n");
syms u
r_eq = subs(r, [cos(theta3) sin(theta3)], [(1-u^2)/(1+u^2) 2*u/(1+u^2)]) == px^2 + py^2 + pz^2;
u_eq_theta3 = simplify(solve(r_eq, u));
disp("u_eq_theta3 = ");
disp(u_eq_theta3);
theta3 = "atan(u_eq_theta3) * 2";
disp("theta3 = ");
disp(theta3);

% theta2求解
fprintf("\n\n****************[Solving theta2]****************\n");
z_eq = pz == P0_4_ORG(3);
z_eq = simplify(subs(z_eq, g3, g(3)));
z_eq = subs(z_eq, [cos(theta2) sin(theta2)], [(1-u^2)/(1+u^2) 2*u/(1+u^2)]);
z_eq = subs(z_eq, [f1 f2], [f(1) f(2)]);
u_eq_theta2 = simplify(solve(z_eq, u));
disp("u_eq_theta2 = ");
disp(u_eq_theta2);
theta2 = "atan(u_eq_theta2) * 2";
disp("theta2 = ");
disp(theta2);

% theta1
fprintf("\n\n****************[Solving theta1]****************\n");
x_eq = px == P0_4_ORG(1);
x_eq = subs(x_eq, [cos(theta1) sin(theta1)], [(1-u^2)/(1+u^2) 2*u/(1+u^2)]);
x_eq = subs(x_eq, [g1 g2], [g(1) g(2)]);
x_eq = subs(x_eq, [f1 f2 f3], [f(1) f(2) f(3)]);

y_eq = py == P0_4_ORG(2);
y_eq = subs(y_eq, [cos(theta1) sin(theta1)], [(1-u^2)/(1+u^2) 2*u/(1+u^2)]);
y_eq = subs(y_eq, [g1 g2], [g(1) g(2)]);
y_eq = subs(y_eq, [f1 f2 f3], [f(1) f(2) f(3)]);

u_eq_theta1_solve = solve(x_eq, u);
u_eq_theta1 = simplify(u_eq_theta1_solve);
disp("u_eq_theta1 = ");
disp(u_eq_theta1);
disp("condition: The equation y_eq holds")
disp("y_eq = ")
disp(y_eq)
theta1 = "atan(u_eq_theta1) * 2";
disp("theta1 = ");
disp(theta1);

% 欧拉角z-y-z求解
% 直接求出来的theta4_zyz\theta5_zyz\theta6_zyz需要转化为D-H坐标下的theta4\5\6
T0_4 = T0_1 * T1_2 * T2_3 * T3_4;
R0_4 = T0_4(1:3, 1:3);
R0_6 = T_target(1:3, 1:3);
R4_6 = inv(R0_4) * R0_6;

% theta5
fprintf("\n\n****************[Solving theta5]****************\n");
r31 = R4_6(3,1);
r32 = R4_6(3,2);
r33 = R4_6(3,3);
disp("r31 =");
disp(simplify(r31));
disp("r32 =");
disp(simplify(r32));
disp("r33 =");
disp(simplify(r33));
theta5_zyz = "atan2(sqrt(r31^2 + r32^2), r33)";
disp("theta5_zyz =");
disp(theta5_zyz);
disp("theta5 = -theta5_zyz + pi");

% theta4
fprintf("\n\n****************[Solving theta4]****************\n");
disp("if theta5 == 0, theta4 = 0, theta6 = atan2(-r12, r11)")
disp("if theta5 == 180, theta4 = 0, theta6 = atan2(-r12, r11)")
r23 = R4_6(2,3);
r13 = R4_6(1,3);
disp("r23 =");
disp(simplify(r23));
disp("r13 =");
disp(simplify(r13));
theta4 = "atan2(r23, r13)";
disp("theta4 = theta4_zyz = ");
disp(theta4);
disp("if deg_theta5_zyz == 0 || deg_theta5_zyz == 180, theta4_zyz = 0")

% theta6
fprintf("\n\n****************[Solving theta6]****************\n");
r32 = R4_6(3,2);
r31 = R4_6(3,1);
r12 = R4_6(1,2);
r11 = R4_6(1,1);
disp("r32 =");
disp(simplify(r32));
disp("r31 =");
disp(simplify(r31));
theta6_zyz = "atan2(r32, -r31) + sign(theta6)*pi";
disp("theta6_zyz = ");
disp(theta6_zyz);
disp("theta6 = theta6_zyz -pi");
disp("if deg_theta5_zyz == 0 || deg_theta5_zyz == 180, theta6_zyz = atan2(-r12, r11)")
disp("r12 = ");
disp(simplify(r12));
disp("r11 = ");
disp(simplify(r11));
disp("")

