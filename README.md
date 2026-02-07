# 零一造物_ZERO机械臂
<img src="./4. Other/Images/model.jpg" width="350" height="350">

### 废话
大家好，我是零一造物，很高兴能在这里与大家相遇！说来惭愧，写代码那么多年，还是第一次开源🤦‍♂️，有点小紧张，哈哈哈哈哈。不管怎么样，真心地希望我能给大家带来帮助。

### 介绍
B站: [【开源】为了跟上大佬们的步伐，我决定从零到一做个智能机械臂](https://www.bilibili.com/video/BV1d63Dz7ERN/?spm_id_from=333.337.search-card.all.click&vd_source=7251eed0b98da55ac3a9186aef37417a)  
小红书：[【开源】为了跟上大佬们的步伐，我决定从零到一做个智能机械臂](https://www.xiaohongshu.com/discovery/item/6868cb63000000002203e9a4?source=webshare&xhsshare=pc_web&xsec_token=ABQKk3qCxYMCLeP3KhxZo0l8gZridNY4JZmONQEwuVI18=&xsec_source=pc_share)  
抖音: [【开源】为了跟上大佬们的步伐，我决定从零到一做个智能机械臂](https://v.douyin.com/LI0cadQi1AU/)  
(广告先做起来😄)  

机械臂ZERO是我在机器人领域的第一次探索，这个项目的诞生源于一个简单的愿望——打造一款高性价比、人人都能拥有的桌面级机械臂。我希望通过它，将机器人领域的核心知识串联起来，从基础的结构设计、运动学原理、控制理论，到前沿的强化学习、大模型等方向，逐步构建一个完整的学习体系。  

更重要的是，我想把这段学习旅程分享给所有志同道合的朋友，和大家一起交流想法、碰撞灵感，甚至合作探索更酷的可能性。如果这个项目能为你带来一点点启发或帮助，那将是我最大的快乐！  

让我们一起学习，一起进步，一起从零到一！

本项目计划完成如下功能：
- 机械臂结构设计 (已完成，2025-07-09)
- 机器人运动学逆解 (已完成，2025-07-09)
- PID控制 (已完成，2025-07-09)
- 串口控制 （已完成，2025-07-09）
- 手柄控制 （已完成，2025-07-09）
- MQTT远程控制 
- WEB可视化平台
- 语音识别
- 语音合成
- 大模型接入
- 数据收集及强化学习 （已完成，2025-10-21）
____
### 目录结构
1.  Model: 机械臂三维模型文件
2.  Software:  存放所有代码类文件  
    2.1. robot: 机械臂嵌入式控制代码
3.  Simulink： 运动学仿真模型
4.  BOM: 物料清单
5.  Deep LR: 深度强化学习
____
### 物料清单
所有物料的型号、数量以及购买链接已全部整理到BOM.xlsx中，整个机械臂的物料成本仅需1300左右。
____
### 结构设计
本机械臂结构采用SolidWorks 2022进行三维建模，推荐大家使用同版本的软件，以便获得完整的设计体验。该软件支持运动仿真，且易于二次开发，我还贴心地给大家转了step文件，放在Model/step目录下。
(此外偷懒没有画螺丝，大家用工具量一下孔洞大小和长度自己匹配一下吧😝)

#### 需要3D打印的模型全部集中放在Model/3d_printing_all.3mf中！！！
#### 安装教程详见Model/installation_guide.md！！！
____
### 软件开发
#### 嵌入式软件
机械臂主控芯片采用STM32F407，开发工具为STM32官方两兄弟STM32CubeMX和STM32CubeIDE。
STM32CubeMX用于硬件外设初始化, STM32CubeIDE我一般用来DEBUG和程序下载。敲代码我还是喜欢用VSCode，那两兄弟就是工具人🐶。

核心功能代码包括: 
1. SoftWare/robot/Core/robot.c 机械臂主要控制逻辑  
2. SoftWare/robot/Core/robot_kinematics.c 机械臂运动学逆解算法  
3. SoftWare/robot/Core/robot_cmd.c 机械臂支持的串口/网络命令

关键技术栈包括：
1.  C语言开发
2.  FreeRTOS操作系统
3.  电机与开发板CAN协议通讯
4.  机械臂运动学逆解
5.  PID控制
6.  MQTT网络协议

串口命令

|    命令    |    参数    |    功能    |
| :--------: | :--------: | :--------: |
|      rel_rotate  |      joint_id angle  |   指定关节的旋转指定角度, 支持负数   |
|      hard_reset  |      \  |   硬复位：各关节复位, 直到限位开关触发   |
|      soft_reset  |      \  |   软复位：各关节复位至零点   |
|      zero  |      \  |   将当前位置设置为零点，软复位将基于当前位置进行复位   |
|      auto  |      x y z  |   机械臂末端移动到指定位置(x,y,z), 注意坐标方向   |

具体参考：SoftWare/robot/Core/robot_cmd.c的robot_uart1_cmd_table数组定义
____
### 硬件链接
<img src="./4. Other/Images/circuit_link.jpg" width="400" height="350">

<img src="./4. Other/Images/mcu.jpg" width="400" height="350">

____
### 仿真模型
本项目的运动学逆解推理和仿真验证均使用MATLAB R2023a。
核心文件包括：
1. Simulink/robot_kinematics_sym_v3_0.m 机械臂运动学逆解符号推导过程 
2. Simulink/model_v1.1/URDF_XG_Robot_Arm_Urdf_V1_1 机械臂urdf文件
3. Simulink/URDF_XG_Robot_Arm_Urdf_Control_V3.slx 机械臂运动学逆解simscape仿真验证模型
4. Simulink/robot_run.m URDF_XG_Robot_Arm_Urdf_Control_V3.slx的一键启动脚本
<img src="./4. Other/Images/simulink.jpg" width="620" height="350">

### 深度强化学习
本项目使用了 TD3 深度强化学习算法控制机械臂的末端跟踪随机目标位置。物理仿真环境使用Mujoco, 强化学习框架使用Stable-Baselines3，当然我自己也实现了一个裸写了一个TD3算法，大家可自行参考（没验证过哦~, 但我感觉应该没啥问题, 哈哈哈哈）
核心文件包括：
1. Deep_RL/robot_arm_env.py 环境惩罚规则代码
2. Deep_RL/train_robot_arm.py 训练代码
3. Deep_RL/robot_arm_mujoco.xml mujoco模型定义
4. Deep_RL/mtd3_robot_arm.ipynb 我写的td3算法😄

训练指令：python train_robot_arm.py
测试指令：python train_robot_arm.py --test --model-path ./logs/best_model/best_model.zip --normalize-path ./logs/best_model/vec_normalize.pkl --episodes 50
<img src="./4. Other/Images/RL.jpg" width="620" height="350">
____
### 待改进点
1. 行星减速器旷量较大, 存在晃动。（还是买太便宜了😝）
2. 虽然安装了限位开关用于初始位置定位，但上电时系统无法自动判断复位运动方向，导致复位过程存在不确定性。
3. 电路板\电路裸露在外。
4. 存3D打印连接件，在负载场景下，存在刚性不足的问题。

**这一代机械臂主要以学习为主，性价比高，通过它基本能串联起机器人领域的核心知识，当然有诸多待改进的地方，欢迎大家一起找bug，并向我们提交！**   
~~**下一代机械臂精度和稳定性都做了升级，加入了更多工业级设计，外观也更赛博（当然价格也小幅上升🐶），如果对性能有要求的小伙伴，可以关注我们第二代机械臂ZERO开发计划！**~~（计划搁置！up准备做四足或者人形机器人了哈哈哈~）


### 交流群
欢迎大家加入我们的技术交流群！🤖  
QQ群：1041684044（已扩充 2000 人，快冲！）   
QQ2群：1029351597