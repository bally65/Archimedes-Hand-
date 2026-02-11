import struct
import os

def ascii_to_binary_stl(ascii_path, binary_path):
    try:
        with open(ascii_path, 'r') as f:
            lines = f.readlines()

        if not lines[0].strip().startswith('solid'):
            print(f"⏩ {ascii_path} is already binary or not standard ASCII STL. Skipping.")
            return

        facets = []
        current_facet = None
        for line in lines:
            line = line.strip()
            if line.startswith('facet normal'):
                current_facet = {'normal': [float(x) for x in line.split()[2:]], 'vertices': []}
            elif line.startswith('vertex'):
                current_facet['vertices'].append([float(x) for x in line.split()[1:]])
            elif line.startswith('endfacet'):
                facets.append(current_facet)

        with open(binary_path, 'wb') as f:
            f.write(b'\x00' * 80) # Header
            f.write(struct.pack('<I', len(facets)))
            for facet in facets:
                f.write(struct.pack('<fff', *facet['normal']))
                for vertex in facet['vertices']:
                    f.write(struct.pack('<fff', *vertex))
                f.write(struct.pack('<H', 0))
        print(f"✅ Converted {ascii_path} to binary.")
    except Exception as e:
        print(f"❌ Error converting {ascii_path}: {e}")

if __name__ == "__main__":
    base_dir = "/home/aa598/.openclaw/workspace/robotics/archimedes-hand/5. Deep_LR/meshes"
    ascii_to_binary_stl(os.path.join(base_dir, "base_plate.stl"), os.path.join(base_dir, "base_plate.stl"))
    ascii_to_binary_stl(os.path.join(base_dir, "screw_left.stl"), os.path.join(base_dir, "screw_left.stl"))
    ascii_to_binary_stl(os.path.join(base_dir, "screw_right.stl"), os.path.join(base_dir, "screw_right.stl"))
