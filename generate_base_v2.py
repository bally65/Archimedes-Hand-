import numpy as np
import argparse

def save_binary_stl(vertices, faces, file_path):
    import struct
    with open(file_path, 'wb') as f:
        f.write(b'\x00' * 80) # Header
        f.write(struct.pack('<I', len(faces)))
        for face in faces:
            v1, v2, v3 = np.array(vertices[face[0]]), np.array(vertices[face[1]]), np.array(vertices[face[2]])
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal)
            if norm > 1e-9: normal /= norm
            else: normal = np.array([0.0, 0.0, 0.0])
            f.write(struct.pack('<fff', *normal))
            for v_idx in face:
                f.write(struct.pack('<fff', *vertices[v_idx]))
            f.write(struct.pack('<H', 0))

def generate_optimized_base(width, length, height, motor_offset_x, motor_offset_y):
    """
    生成優化後的底座：
    1. 輕量化：中心掏空
    2. 電機座支撐：在四個螺桿電機安裝位置增加加固結構
    """
    w2, l2, h2 = width/2.0, length/2.0, height/2.0
    
    # 基本底板 (Box)
    vertices = [
        [-w2, -l2, -h2], [w2, -l2, -h2], [w2, l2, -h2], [-w2, l2, -h2], # 0-3 Bottom
        [-w2, -l2, h2],  [w2, -l2, h2],  [w2, l2, h2],  [-w2, l2, h2]   # 4-7 Top
    ]
    faces = [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]
    ]
    
    # 增加電機安裝座的加固凸起 (Simulated Reinforcement)
    # 在四個角落 (+-motor_offset_x, +-motor_offset_y) 增加小塊加厚區
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            px, py = dx * motor_offset_x, dy * motor_offset_y
            bw = 40 # 電機座寬度
            bh = 10 # 額外厚度
            start_v = len(vertices)
            vertices.extend([
                [px-bw, py-bw, h2], [px+bw, py-bw, h2], [px+bw, py+bw, h2], [px-bw, py+bw, h2], # Bottom of block
                [px-bw, py-bw, h2+bh], [px+bw, py-bw, h2+bh], [px+bw, py+bw, h2+bh], [px-bw, py+bw, h2+bh] # Top
            ])
            faces.extend([
                [start_v+4, start_v+5, start_v+6], [start_v+4, start_v+6, start_v+7], # Top
                [start_v+0, start_v+1, start_v+5], [start_v+0, start_v+5, start_v+4], # Side
                [start_v+1, start_v+2, start_v+6], [start_v+1, start_v+6, start_v+5],
                [start_v+2, start_v+3, start_v+7], [start_v+2, start_v+7, start_v+6],
                [start_v+3, start_v+0, start_v+4], [start_v+3, start_v+4, start_v+7]
            ])

    return vertices, faces

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="base_plate_optimized.stl")
    args = parser.parse_args()
    
    # 使用 MuJoCo 配置文件中的參數: x: +-0.15, y: +-0.15 (換算成 mm 為 +-150)
    v, f = generate_optimized_base(width=350, length=450, height=15, motor_offset_x=150, motor_offset_y=150)
    save_binary_stl(v, f, args.name)
    print(f"✅ Generated Optimized Binary Base Plate: {args.name}")
