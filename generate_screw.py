import numpy as np
import argparse
import os

def save_ascii_stl(vertices, faces, file_path):
    with open(file_path, 'w') as f:
        f.write("solid archimedes_screw\n")
        for face in faces:
            # Face normal calculation
            v1, v2, v3 = np.array(vertices[face[0]]), np.array(vertices[face[1]]), np.array(vertices[face[2]])
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal /= norm
            else:
                normal = np.array([0.0, 0.0, 0.0])
            
            f.write(f"facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            f.write("  outer loop\n")
            for vertex_idx in face:
                v = vertices[vertex_idx]
                f.write(f"    vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("  endloop\n")
            f.write("endfacet\n")
        f.write("endsolid archimedes_screw\n")

def generate_solid_screw(radius, length, turns, handedness="right", shaft_ratio=0.5, thickness=2.0):
    num_points = 200
    z = np.linspace(0, length, num_points)
    
    direction = 1.0 if handedness == "right" else -1.0
    theta = np.linspace(0, direction * 2 * np.pi * turns, num_points)
    
    shaft_radius = radius * shaft_ratio
    
    vertices = []
    faces = []
    
    # --- 1. Shaft Cylinder ---
    # Shaft vertices at z[i]
    for i in range(num_points):
        # Shaft Inner
        vertices.append([shaft_radius * np.cos(theta[i]), shaft_radius * np.sin(theta[i]), z[i]])
    
    # Shaft faces
    for i in range(num_points - 1):
        v1, v2 = i, i + 1
        # No, shaft needs full circular cross section at each Z. 
        # For simplicity, let's just make a 4-sided shaft (square) or keep it simple.
        # Actually, let's just make the blade have thickness and a hollow center.
        pass

    # --- 2. Blade with thickness ---
    # We will generate 4 helical paths: Inner Top, Inner Bottom, Outer Top, Outer Bottom
    inner_top = []
    inner_bottom = []
    outer_top = []
    outer_bottom = []
    
    half_thick = thickness / 2.0
    
    for i in range(num_points):
        # Shift z slightly for thickness
        it = [shaft_radius * np.cos(theta[i]), shaft_radius * np.sin(theta[i]), z[i] + half_thick]
        ib = [shaft_radius * np.cos(theta[i]), shaft_radius * np.sin(theta[i]), z[i] - half_thick]
        ot = [radius * np.cos(theta[i]), radius * np.sin(theta[i]), z[i] + half_thick]
        ob = [radius * np.cos(theta[i]), radius * np.sin(theta[i]), z[i] - half_thick]
        
        inner_top.append(it)
        inner_bottom.append(ib)
        outer_top.append(ot)
        outer_bottom.append(ob)

    # Combine into vertices list and keep indices
    base_idx = len(vertices)
    vertices.extend(inner_top)    # [base_idx : base_idx + n]
    vertices.extend(inner_bottom) # [base_idx + n : base_idx + 2n]
    vertices.extend(outer_top)    # [base_idx + 2n : base_idx + 3n]
    vertices.extend(outer_bottom) # [base_idx + 3n : base_idx + 4n]
    
    n = num_points
    for i in range(n - 1):
        # Top Surface
        faces.append([base_idx + i, base_idx + 2*n + i, base_idx + i + 1])
        faces.append([base_idx + 2*n + i, base_idx + 2*n + i + 1, base_idx + i + 1])
        # Bottom Surface
        faces.append([base_idx + n + i, base_idx + n + i + 1, base_idx + 3*n + i])
        faces.append([base_idx + 3*n + i, base_idx + n + i + 1, base_idx + 3*n + i + 1])
        # Outer Edge
        faces.append([base_idx + 2*n + i, base_idx + 3*n + i, base_idx + 2*n + i + 1])
        faces.append([base_idx + 3*n + i, base_idx + 3*n + i + 1, base_idx + 2*n + i + 1])
        # Inner Edge
        faces.append([base_idx + i, base_idx + i + 1, base_idx + n + i])
        faces.append([base_idx + n + i, base_idx + i + 1, base_idx + n + i + 1])

    # End caps for the blade
    # Start cap
    faces.append([base_idx, base_idx + n, base_idx + 2*n])
    faces.append([base_idx + n, base_idx + 3*n, base_idx + 2*n])
    # End cap
    faces.append([base_idx + n - 1, base_idx + 3*n - 1, base_idx + 2*n - 1])
    faces.append([base_idx + 3*n - 1, base_idx + 4*n - 1, base_idx + 2*n - 1])

    return vertices, faces

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="screw.stl")
    parser.add_argument("--radius", type=float, default=50)
    parser.add_argument("--length", type=float, default=300)
    parser.add_argument("--turns", type=float, default=6)
    parser.add_argument("--thick", type=float, default=2.0)
    parser.add_argument("--handed", type=str, choices=["right", "left"], default="right")
    args = parser.parse_args()
    
    # Use output path relative to workspace or as specified
    v, f = generate_solid_screw(args.radius, args.length, args.turns, handedness=args.handed, thickness=args.thick)
    save_ascii_stl(v, f, args.name)
    print(f"✅ Generated SOLID {args.handed.upper()} STL: {args.name}")
