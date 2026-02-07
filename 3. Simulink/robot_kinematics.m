clc   % 清除命令窗口的内容，对工作环境中的全部变量无任何影响
% 机械臂运动学逆解析
disp(D_H)

% 读取a,alpha下标3,应该为第4个元素，故删除第一个元素，方便读取
% alpha为角度，提前除以pi，方便使用sinpi\cospi提高精度
a = D_H(:,1);
a = a(2:end);
alpha = D_H(:,2);
alpha = alpha(2:end)/pi;
d = D_H(:,3);
trans = [0 -47.63 15.5]; % 初始状态作为验证
r_value = sqrt(trans(1)^2 + trans(2)^2 + trans(3)^2);

syms c3 s3 c2 s2 u r z theta3 theta2
f1 = a(3)*c3 + d(4)*sinpi(alpha(3))*s3 + a(2);
f2 = a(3)*cospi(alpha(2))*s3 - d(4)*sinpi(alpha(3))*cospi(alpha(2))*c3 - d(4)*sinpi(alpha(2))*cospi(alpha(3)) - d(3)*sinpi(alpha(2));
f3 = a(3)*sinpi(alpha(2))*s3 - d(4)*sinpi(alpha(3))*sinpi(alpha(2))*c3 + d(4)*cospi(alpha(2))*cospi(alpha(3)) + d(3)*cospi(alpha(2));
k3 = f1^2 + f2^2 + f3^2 + a(1)^2 + d(2)^2 + 2*d(2)*f3;
k1 = f1;
k2 = -f2;
k4 = f3*cospi(alpha(1)) + d(2)*cospi(alpha(1));
disp("k3 = ")
disp(k3)

% 求出theta3
k3 = subs(k3, c3, (1 - u^2)/(1 + u^2));
k3 = subs(k3, s3, 2*u/(1 + u^2));
disp("k3(u) = ")
disp(k3);
solution3 = solve(r == k3, u);
disp(solution3);
solution3 = subs(solution3, r, r_value);
disp(solution3);





% % 已求出theta3, 求theta2
% equation2 = z == (k1*s2 - k2*c2)*sinpi(alpha(1)) + k4;
% 
% % u = tan(theta2/2)
% equation2 = subs(equation2, c2, (1 - u^2)/(1 + u^2));
% equation2 = subs(equation2, s2, 2*u/(1 + u^2));
% equation2 = subs(equation2, u, tan(theta2/2));
% solution2 = solve(equation2, theta2);
% disp('theta2 = ')
% disp(solution2)
% 
% % 已求出theta3\theta2, 求theta1
% syms g1 g2 g3 x c1 s1 theta1
% g1 = c2*f1 - s2*f2 + a(1);
% g2 = s2*cospi(alpha(1))*f1 + c2*cospi(alpha(1))*f2 - sinpi(alpha(1))*f3 - d(2)*sinpi(alpha(1));
% g3 = s2*sinpi(alpha(1))*f1 + c2*sinpi(alpha(1))*f2 +cospi(alpha(1))*f3 + d(2)*cospi(alpha(1));
% equation1 = x == c1*g1 - s1*g2;
% % u = tan(theta1/2)
% equation1 = subs(equation1, c1, (1 - u^2)/(1 + u^2));
% equation1 = subs(equation1, s1, 2*u/(1 + u^2));
% equation1 = subs(equation1, u, tan(theta1/2));
% % solution1 = solve(equation1, theta1, 'ReturnConditions', true);
% solution1 = solve(equation1, theta1);
% disp('theta1 = ');
% disp(solution1);
% % disp(solution1.theta1)
% % disp(solution1.parameters)
% % disp(solution1.conditions)




