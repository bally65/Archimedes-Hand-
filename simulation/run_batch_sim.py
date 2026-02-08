import os
import subprocess
import time

ENVS = {
    "water": {"model": "Newtonian", "nu": "1e-06"},
    "soft_sludge": {"model": "Bingham", "nu": "0.01", "yieldStress": "10"},
    "medium_mud": {"model": "HerschelBulkley", "nu": "0.1", "yieldStress": "50", "n": "0.8"},
    "heavy_clay": {"model": "HerschelBulkley", "nu": "1.0", "yieldStress": "200", "n": "0.5"}
}

def update_transport_properties(env_name, params):
    content = f"""FoamFile {{ version 2.0; format ascii; class dictionary; object transportProperties; }}
transportModel {params['model']};
nu [0 2 -1 0 0 0 0] {params.get('nu', '1e-06')};
"""
    if params['model'] == "Bingham":
        content += f"BinghamCoeffs {{ yieldStress [1 -1 -2 0 0 0 0] {params['yieldStress']}; }}\n"
    elif params['model'] == "HerschelBulkley":
        content += f"HerschelBulkleyCoeffs {{ yieldStress [1 -1 -2 0 0 0 0] {params['yieldStress']}; n [0 0 0 0 0 0 0] {params['n']}; }}\n"
    
    with open("constant/transportProperties", "w") as f:
        f.write(content)

def run_sim(env_name):
    print(f"🚀 Starting simulation for: {env_name}")
    log_file = f"log.simpleFoam.{env_name}"
    with open(log_file, "w") as f:
        subprocess.run(["simpleFoam"], stdout=f, stderr=f)
    print(f"✅ Finished: {env_name}")

if __name__ == "__main__":
    os.chdir("projects/robotics/simulation/screw_v1")
    for name, params in ENVS.items():
        update_transport_properties(name, params)
        run_sim(name)
