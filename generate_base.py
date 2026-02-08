import numpy as np
import argparse

def save_ascii_stl(vertices, faces, file_path):
    with open(file_path, 'w') as f:
        f.write("solid base_plate\n")
        for face in faces:
            v1, v2, v3 = np.array(vertices[face[0]]), np.array(vertices[face[1]]), np.array(vertices[face[2]])
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal)
            if norm > 1e-9: normal /= norm
            else: normal = np.array([0.0, 0.0, 0.0])
            f.write(f"facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            f.write("  outer loop\n")
            for vertex_idx in face:
                v = vertices[vertex_idx]
                f.write(f"    vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("  endloop\n")
            f.write("endfacet\n")
        f.write("endsolid base_plate\n")

def generate_base_plate(width, length, height):
    # Simple box
    w2, l2, h2 = width/2.0, length/2.0, height/2.0
    vertices = [
        [-w2, -l2, -h2], [w2, -l2, -h2], [w2, l2, -h2], [-w2, l2, -h2], # Bottom
        [-w2, -l2, h2],  [w2, -l2, h2],  [w2, l2, h2],  [-w2, l2, h2]   # Top
    ]
    faces = [
        [0, 2, 1], [0, 3, 2], # Bottom
        [4, 5, 6], [4, 6, 7], # Top
        [0, 1, 5], [0, 5, 4], # Front
        [1, 2, 6], [1, 6, 5], # Right
        [2, 3, 7], [2, 7, 6], # Back
        [3, 0, 4], [3, 4, 7]  # Left
    ]
    return vertices, faces

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="base_plate.stl")
    parser.add_argument("--w", type=float, default=250)
    parser.add_argument("--l", type=float, default=400)
    parser.add_argument("--h", type=float, default=20)
    args = parser.parse_args()
    
    v, f = generate_base_plate(args.w, args.l, args.h)
    save_ascii_stl(v, f, args.name)
    print(f"✅ Generated Base Plate: {args.name}")
