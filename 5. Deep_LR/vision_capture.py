import mujoco
import numpy as np
from PIL import Image
import os

def capture_simulation_frame(xml_path, output_path="scene_snapshot.png"):
    """
    Captures a high-quality frame from the current MuJoCo scene for Vision-LLM analysis.
    """
    os.environ['MUJOCO_GL'] = 'osmesa' # Use software rendering for headless environments
    print(f"📸 Capturing MuJoCo frame (OSMesa) from {xml_path}...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Offscreen rendering context
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Step simulation slightly to stabilize
    mujoco.mj_step(model, data)
    
    # Update renderer
    renderer.update_scene(data)
    pixels = renderer.render()
    
    # Save image
    img = Image.fromarray(pixels)
    img.save(output_path)
    print(f"✅ Snapshot saved to {output_path}")

if __name__ == "__main__":
    xml = "/home/aa598/.openclaw/workspace/robotics/archimedes-hand/5. Deep_LR/archimedes_hand_mujoco.xml"
    capture_simulation_frame(xml)
