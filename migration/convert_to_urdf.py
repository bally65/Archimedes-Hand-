import xml.etree.ElementTree as ET
import os

def mujoco_to_urdf_skeleton(mjcf_path, urdf_path):
    """
    Creates a basic URDF skeleton from a MuJoCo XML file.
    Note: Complex joints and specific MuJoCo physics won't map 1:1, but meshes will.
    """
    print(f"🛠️ Converting {mjcf_path} to URDF skeleton...")
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    
    urdf = ET.Element("robot", name="archimedes_hand")
    
    # Simple mapping of bodies to links
    for body in root.findall(".//body"):
        name = body.get("name")
        link = ET.SubElement(urdf, "link", name=name)
        
        # Visual/Collision mapping
        for geom in body.findall("geom"):
            visual = ET.SubElement(link, "visual")
            origin = ET.SubElement(visual, "origin", xyz=geom.get("pos", "0 0 0"), rpy="0 0 0")
            geometry = ET.SubElement(visual, "geometry")
            if geom.get("type") == "mesh":
                mesh_name = geom.get("mesh")
                # Look up mesh file in assets/compiler
                ET.SubElement(geometry, "mesh", filename=f"package://meshes/{mesh_name}.STL")
    
    # Save URDF
    output_xml = ET.tostring(urdf, encoding='unicode')
    with open(urdf_path, "w") as f:
        f.write(output_xml)
    print(f"✅ URDF Skeleton saved to {urdf_path}")

if __name__ == "__main__":
    xml = "/home/aa598/.openclaw/workspace/robotics/archimedes-hand/5. Deep_LR/archimedes_hand_mujoco.xml"
    out = "/home/aa598/.openclaw/workspace/robotics/archimedes-hand/migration/archimedes_hand.urdf"
    mujoco_to_urdf_skeleton(xml, out)
