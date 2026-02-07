import pygame
import serial
import time
import argparse

def send_and_receive_joystick_data(port, baudrate):
    # 初始化pygame和手柄
    pygame.init()
    pygame.joystick.init()
    # 创建一个 Pygame 显示窗口
    screen = pygame.display.set_mode((600, 200))
    pygame.display.set_caption("Joystick Control")

    # 设置字体和文字内容
    font = pygame.font.Font(None, 36)
    text = font.render("Robot joystick is running, press Q to exit", True, (255, 255, 255))
    text_rect = text.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2))

    # 检查是否有手柄连接
    if pygame.joystick.get_count() == 0:
        print("No joystick connected!")
        pygame.quit()
        return

    # 获取第一个手柄
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # 打开指定的串口
    ser = None  # 初始化串口对象
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Serial port {port} opened successfully at baudrate {baudrate}.")
        # 发送remote enable命令
        ser.write(b"remote_enable\n")
        time.sleep(3)
    except serial.SerialException as e:
        print(f"Failed to open serial port {port}: {e}")
        pygame.quit()
        return

    # 定义一个函数来获取手柄的axes状态
    def get_joystick_axes():
        pygame.event.pump()  # 处理pygame事件队列
        # 获取所有轴的值，并保留小数点后两位
        axes = [round(joystick.get_axis(i), 2) for i in range(joystick.get_numaxes())]
        return axes

    # 主循环
    try:
        while True:
            axes = get_joystick_axes()
            # 将axes状态转换为指定格式的字符串
            # 格式为：remote_event 0.0 -0.0 0.0 -0.0 -1.0 -1.0
            axes_str = "remote_event " + " ".join(map(str, axes)) + "\n"
            # 发送状态到串口
            ser.write(axes_str.encode())
            # print(f"Sent to {port}: {axes_str.strip()}")

            # 接收串口返回的数据
            if ser.in_waiting > 0:
                received_data = ser.readline().decode().strip()
                print(f"Received from {port}: {received_data}")

            # 处理键盘事件
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # 检测 Q 键是否被按下
                        if ser and ser.is_open:
                            ser.write(b"remote_disable\n")
                        pygame.quit()
                        return

            # 填充背景色
            screen.fill((0, 0, 0))
            # 将文字绘制到屏幕上
            screen.blit(text, text_rect)
            # 等待100ms
            time.sleep(0.1)
            # 更新显示
            pygame.display.flip()
    finally:
        # 关闭串口和pygame
        if ser and ser.is_open:
            ser.close()
        pygame.quit()

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Send joystick data to a serial port and receive data back.")
    parser.add_argument("-p", "--port", type=str, default="COM3", help="Serial port (default: COM3)")
    parser.add_argument("-b", "--baudrate", type=int, default=115200, help="Baudrate (default: 115200)")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数
    send_and_receive_joystick_data(args.port, args.baudrate)