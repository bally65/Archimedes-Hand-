# Archimedean Screw Simulation Implementation Plan

> **For Molty:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simulate thrust and lift of a rotating Archimedean screw in a fluid environment using OpenFOAM to validate propulsion for the Zero Arm mobile base.

**Architecture:** Use `blockMesh` to define the domain, `snappyHexMesh` to integrate the screw STL, and `simpleFoam` (with Rotating Reference Frame or MRF) to simulate steady-state fluid forces.

**Tech Stack:** OpenFOAM (1912), Python (STL pre-processing), Bash.

---

### Task 1: STL Cleanup and Orientation

**Files:**
- Modify: `projects/garage/generate_screw.py`
- Create: `projects/robotics/simulation/screw_v1/constant/triSurface/screw.stl`

**Step 1: Update generator to create closed manifolds**
Update `generate_screw.py` to add caps and thickness so it's a solid, not a ribbon.

**Step 2: Generate and copy to simulation folder**
Run the script and copy the output to the OpenFOAM surface directory.

**Step 3: Commit**
`git add projects/garage/generate_screw.py projects/robotics/simulation/screw_v1/constant/triSurface/screw.stl && git commit -m "sim: prepare solid screw STL for OpenFOAM"`

### Task 2: Domain Definition (blockMesh)

**Files:**
- Create: `projects/robotics/simulation/screw_v1/system/blockMeshDict`

**Step 1: Define a cylindrical or rectangular domain**
Create a bounding box large enough to avoid boundary interference.

**Step 2: Run blockMesh**
Run `blockMesh` in the simulation directory.

**Step 3: Commit**
`git add projects/robotics/simulation/screw_v1/system/blockMeshDict && git commit -m "sim: define simulation domain"`

### Task 3: Surface Refinement (snappyHexMesh)

**Files:**
- Create: `projects/robotics/simulation/screw_v1/system/snappyHexMeshDict`
- Create: `projects/robotics/simulation/screw_v1/system/meshQualityDict`

**Step 1: Configure snappyHexMesh**
Set refinement levels for the screw surface.

**Step 2: Run snappyHexMesh**
Run `snappyHexMesh -overwrite`.

**Step 3: Commit**
`git add projects/robotics/simulation/screw_v1/system/snappyHexMeshDict && git commit -m "sim: refine mesh around screw"`

### Task 4: Physics and Boundary Conditions

**Files:**
- Create: `projects/robotics/simulation/screw_v1/0/U`
- Create: `projects/robotics/simulation/screw_v1/0/p`
- Create: `projects/robotics/simulation/screw_v1/constant/transportProperties`

**Step 1: Set velocity and pressure**
Define inlet/outlet and the rotating wall condition for the screw.

**Step 2: Commit**
`git add projects/robotics/simulation/screw_v1/0/ projects/robotics/simulation/screw_v1/constant/transportProperties && git commit -m "sim: set BCs and fluid properties"`

### Task 5: Force Extraction (functionObjects)

**Files:**
- Modify: `projects/robotics/simulation/screw_v1/system/controlDict`

**Step 1: Add forces function object**
Configure OpenFOAM to calculate $F_x$ (Thrust) and $F_z$ (Lift) during the run.

**Step 2: Commit**
`git add projects/robotics/simulation/screw_v1/system/controlDict && git commit -m "sim: add force extraction logic"`
