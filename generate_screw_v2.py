import numpy as np
import argparse
import os
import struct

def save_binary_stl(vertices, faces, file_path):
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

def generate_industrial_screw(radius, length, turns, handedness="right", thickness=0.003, tapering=0.1):
    """
    Optimized Screw Design (SI Units: Meters):
    1. Tapered Ends: Reduces drag.
    2. Integrated Shaft: Stronger physics.
    """
    num_radial = 16
    num_z = 100
    z_vals = np.linspace(0, length, num_z)
    direction = 1.0 if handedness == "right" else -1.0
    theta_vals = np.linspace(0, direction * 2 * np.pi * turns, num_z)
    vertices = []
    faces = []
    shaft_r = radius * 0.4
    for i in range(num_z):
        for r_step in range(num_radial):
            angle = (2 * np.pi * r_step) / num_radial
            vertices.append([shaft_r * np.cos(angle), shaft_r * np.sin(angle), z_vals[i]])
    for i in range(num_z - 1):
        for j in range(num_radial):
            curr, next_r = i * num_radial + j, i * num_radial + (j + 1) % num_radial
            below, below_next = (i + 1) * num_radial + j, (i + 1) * num_radial + (j + 1) % num_radial
            faces.append([curr, next_r, below]); faces.append([next_r, below_next, below])
    blade_start_idx = len(vertices)
    half_thick = thickness / 2.0
    for i in range(num_z):
        scale = 1.0
        if z_vals[i] < length * tapering: scale = z_vals[i] / (length * tapering)
        elif z_vals[i] > length * (1.0 - tapering): scale = (length - z_vals[i]) / (length * tapering)
        current_r = shaft_r + (radius - shaft_r) * scale
        cos_t, sin_t = np.cos(theta_vals[i]), np.sin(theta_vals[i])
        vertices.append([shaft_r * cos_t, shaft_r * sin_t, z_vals[i] + half_thick])
        vertices.append([shaft_r * cos_t, shaft_r * sin_t, z_vals[i] - half_thick])
        vertices.append([current_r * cos_t, current_r * sin_t, z_vals[i] + half_thick])
        vertices.append([current_r * cos_t, current_r * sin_t, z_vals[i] - half_thick])
    for i in range(num_z - 1):
        b, n = blade_start_idx + i * 4, blade_start_idx + (i + 1) * 4
        faces.extend([[b+0, b+2, n+0], [b+2, n+2, n+0], [b+1, n+1, b+3], [b+3, n+1, n+3],
                      [b+2, b+3, n+2], [b+3, n+3, n+2], [b+0, n+0, b+1], [n+0, n+1, b+1]])
    return vertices, faces

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--handed", type=str, choices=["right", "left"], default="right")
    args = parser.parse_args()
    # MuJoCo Scale: Radius 0.06m, Length 0.35m
    v, f = generate_industrial_screw(radius=0.06, length=0.35, turns=5, handedness=args.handed)
    save_binary_stl(v, f, args.name)
    print(f"✅ Generated SI-unit {args.handed.upper()} Screw: {args.name}")
