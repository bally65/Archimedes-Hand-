import mujoco
import numpy as np
import time
import os

# Set relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def test_propulsion():
    try:
        model = mujoco.MjModel.from_xml_path('archimedes_hand_mujoco.xml')
        data = mujoco.MjData(model)
        
        print("🚀 啟動螺旋槳推進測試...")
        
        # 設置旋轉速度 (ctrl 代表扭矩/推力)
        # FL, RL 同向; FR, RR 同向 (抵消橫向力)
        data.actuator('screw_fl_ctrl').ctrl = 10.0
        data.actuator('screw_rl_ctrl').ctrl = 10.0
        data.actuator('screw_fr_ctrl').ctrl = -10.0
        data.actuator('screw_rr_ctrl').ctrl = -10.0
        
        initial_x = data.qpos[0]
        
        # 模擬 2000 步 (約 4 秒)
        for i in range(2000):
            mujoco.mj_step(model, data)
            if i % 500 == 0:
                print(f"   時間: {data.time:.2f}s | 位置 X: {data.qpos[0]:.4f} | 速度 X: {data.qvel[0]:.4f}")
        
        final_x = data.qpos[0]
        displacement = final_x - initial_x
        print(f"✅ 推進測試完成！總位移: {displacement:.4f}m")
        return displacement
    except Exception as e:
        print(f"❌ 測試出錯: {e}")

if __name__ == "__main__":
    test_propulsion()
