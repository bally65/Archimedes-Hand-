function robot_analyse(D_H)
%ROBOT_ANALYSE 显示各link变化矩阵
[rows, ~] = size(D_H);
for i = 1:rows
    row = D_H(i, :); % 获取第i行
    a = row(1);
    alpha = row(2);
    d = row(3);
    fprintf("link%d\n", i);
    fprintf("[cos(t), -sin(t), 0, %.2f;\n", a);
    fprintf("%fsin(t), %fcos(t), %.2f, %.2f;\n", cos(alpha), cos(alpha), -sin(alpha), -d * sin(alpha));
    fprintf("%fsin(t), %fcos(t), %.2f, %.2f;\n", sin(alpha), sin(alpha), cos(alpha), d*cos(alpha));
    fprintf("0, 0, 0, 1]\n\n");
end
end