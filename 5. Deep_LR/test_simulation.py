import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the model
model = mujoco.MjModel.from_xml_path('archimedes_hand_mujoco.xml')
data = mujoco.MjData(model)

def set_screw_velocities(fl, fr, rl, rr):
    """
    Sets the velocities for the 4 screw actuators.
    In MuJoCo, for motor actuators, ctrl represents the torque/force.
    """
    data.actuator('screw_fl_ctrl').ctrl = fl
    data.actuator('screw_fr_ctrl').ctrl = fr
    data.actuator('screw_rl_ctrl').ctrl = rl
    data.actuator('screw_rr_ctrl').ctrl = rr

def set_arm_joints(j1, j2, j3, j4, j5, j6):
    """
    Sets target torques/positions for arm joints.
    Note: Current actuators are 'motor' type, which expect force/torque.
    """
    data.actuator('arm_j1').ctrl = j1
    data.actuator('arm_j2').ctrl = j2
    data.actuator('arm_j3').ctrl = j3
    data.actuator('arm_j4').ctrl = j4
    data.actuator('arm_j5').ctrl = j5
    data.actuator('arm_j6').ctrl = j6

def main():
    # Start the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()

            # --- Simple Test Sequence ---
            elapsed = time.time() - start_time
            
            if elapsed < 5:
                # 1. Forward movement: all screws rotating
                # Screws on the same side rotate in opposite directions to cancel torque,
                # but for linear thrust in mud, we'll simulate a standard pattern.
                set_screw_velocities(5, -5, 5, -5)
                set_arm_joints(0, 0, 0, 0, 0, 0)
            elif elapsed < 10:
                # 2. Stop and move arm
                set_screw_velocities(0, 0, 0, 0)
                # Wave the arm
                j3_val = np.sin(elapsed * 2) * 0.5 + 1.5
                set_arm_joints(0, 0, j3_val, 0, 0, 0)
            else:
                # 3. Rotate in place
                set_screw_velocities(5, 5, 5, 5)

            # Step the simulation
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            # Maintain real-time execution
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
