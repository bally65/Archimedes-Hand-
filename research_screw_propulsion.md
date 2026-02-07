# Research: Archimedean Screw-Propelled Mobile Platform for Zero Arm

## 1. Concept Overview
To enable the Zero Robotic Arm to traverse high-moisture, sediment-rich environments (swamps, mudflats, etc.), a **Screw-Propelled Vehicle (SPV)** chassis is proposed. This utilizes counter-rotating Archimedean screws to generate thrust in media where wheels or tracks would fail.

## 2. Key Advantages for High-Sediment Environments
- **Buoyancy**: Hollow screws can act as pontoons, providing flotation.
- **Traction**: The screw threads "dig" into the mud, converting rotation into linear motion.
- **Self-Cleaning**: The rotating motion tends to shed sticky mud more effectively than track links.

## 3. Design Parameters for the "Archimedes Base"
- **Screw Configuration**: 
    - **Dual-Screw**: Simple, uses differential steering. Requires a stabilizing tail or flat bottom to prevent rolling.
    - **Quad-Screw**: Maximum stability. Allows for lateral (strafing) movement if screws are synchronized correctly.
- **Screw Geometry**:
    - **Pitch**: Higher pitch for speed, lower pitch for higher torque/digging power.
    - **Diameter**: Larger diameter increases buoyancy.
- **Mounting**: The `base_link` of the Zero Arm must be center-mounted to maintain the center of gravity (CoG).

## 4. Technical Challenges
- **Sealing**: High-sediment environments require IP68+ sealing for the motor and drive shafts.
- **Torque**: Moving through thick mud requires high-torque gearboxes (likely planetary reduction, similar to the arm's joints).
- **Steering**: Controlled by varying the relative speed of the left and right screws.

## 5. Integration Plan
1. **CAD Integration**: Import `projects/robotics/1. Model/STEP/base_link.step` into a new assembly with the screw chassis.
2. **Control Logic**: Adapt the Singularity V3 Pattern Recognition to detect surface resistance and adjust screw RPM dynamically.
3. **Simulation**: Use MuJoCo (in the `Deep_LR` folder) to simulate screw-soil interaction.
