import mujoco
import numpy as np
import os

# Set relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    model = mujoco.MjModel.from_xml_path('archimedes_hand_mujoco.xml')
    data = mujoco.MjData(model)
    print("✅ Model loaded successfully!")
    
    # Run 100 steps
    for i in range(100):
        mujoco.mj_step(model, data)
    
    print(f"✅ Simulation stepped 100 times. Robot Z-height: {data.qpos[2]:.4f}")
except Exception as e:
    print(f"❌ Error: {e}")
