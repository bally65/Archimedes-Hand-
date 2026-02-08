# Multi-Environment Simulation Plan: Archimedes' Hand

**Goal:** Test the performance of the Archimedean screw across 4 distinct sediment types to determine optimal torque/RPM settings for each.

## 1. Environment Definitions

| Case ID | Type | OpenFOAM Model | Physical Description |
|---------|------|----------------|----------------------|
| **ENV_W** | Clean Water | Newtonian | Reference baseline (low resistance). |
| **ENV_S** | Soft Sludge | Bingham | High moisture, low yield stress (Swamp). |
| **ENV_M** | Medium Mud | Herschel-Bulkley | Typical riverbank sediment. |
| **ENV_H** | Heavy Clay | Herschel-Bulkley | High viscosity, hard to shear (Drying mud). |

## 2. Parameter Matrix
- **RPM Range**: 100, 200, 300, 400.
- **Pitch Variations**: 1.5cm, 2.0cm, 2.5cm.

## 3. Automation Strategy
I will use a Python wrapper `run_batch_sim.py` to:
1. Modify `constant/transportProperties` for each environment.
2. Execute `simpleFoam`.
3. Extract $F_x$ (Thrust) and $T_z$ (Required Torque).
4. Generate a comparative performance heatmap.

---
*Report will be compiled after all cases complete.* 🧐
