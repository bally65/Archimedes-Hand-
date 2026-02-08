import numpy as np

class ZeroArmIK:
    def __init__(self):
        # DH Parameters for ZERO Arm (approximate based on C code)
        self.a2 = 0.2
        self.a3 = 0.2
        self.d4 = 0.125
        
        self.joint_limits = [
            (-np.pi, np.pi),   # J1
            (-np.pi/2, np.pi/2), # J2
            (0, np.pi),        # J3
            (-np.pi, np.pi),   # J4
            (-np.pi/2, 0),     # J5
            (-np.pi, np.pi)    # J6
        ]

    def solve_ik(self, target_pos, target_quat=None):
        """
        target_pos: [x, y, z]
        Returns: list of 6 joint angles (radians) or None
        Simplified IK implementation based on the C code logic.
        """
        px, py, pz = target_pos
        
        # 1. Solve Theta 3
        a2, a3, d4 = self.a2, self.a3, self.d4
        
        const_eq1 = -a2**4 + 2*a2**2 * (a3**2 + d4**2) - a3**4 - 2*a3**2*d4**2 - d4**4
        const_eq2 = -a2**2 + 2*a2*a3 - a3**2 - d4**2
        pow_distance_2 = px**2 + py**2 + pz**2
        
        eq1_val = (const_eq1 + 2*a2**2*pow_distance_2 + 2*a3**2*pow_distance_2 + 2*d4**2*pow_distance_2 
                   - px**4 - py**4 - pz**4 - 2*px**2*(py**2 + pz**2) - 2*py**2*pz**2)
        
        if eq1_val < 0:
            return None # Unreachable
            
        u_theta3 = -(2*a2*d4 + np.sqrt(eq1_val)) / (const_eq2 + pow_distance_2)
        theta3 = np.arctan(u_theta3) * 2
        
        # 2. Solve Theta 2
        eq1_t2 = np.sqrt(a2**2 + a3**2 + d4**2 + 2*a2*a3*np.cos(theta3) - 2*a2*d4*np.sin(theta3) - pz**2)
        eq2_t2 = a3*np.cos(theta3) - d4*np.sin(theta3)
        eq3_t2 = d4*np.cos(theta3) - pz + a3*np.sin(theta3)
        
        u_theta2 = -(a2 + eq1_t2 + eq2_t2) / eq3_t2
        theta2 = np.arctan(u_theta2) * 2
        
        # 3. Solve Theta 1
        diff_t23 = theta2 - theta3
        eq1_t1 = a2*np.cos(theta2) + a3*np.cos(diff_t23) + d4*np.sin(diff_t23)
        
        # Protect against division by zero or invalid sqrt
        if abs(px + eq1_t1) < 1e-9:
            u_theta1 = 0
        else:
            u_theta1 = np.sqrt(max(0, (-px + eq1_t1)/(px + eq1_t1)))
            
        # Verify sign of u_theta1
        py_calc = (2*u_theta1*(np.cos(theta2)*(a2 + a3*np.cos(theta3) - d4*np.sin(theta3)) + 
                   np.sin(theta2)*(d4*np.cos(theta3) + a3*np.sin(theta3)))) / (u_theta1**2 + 1)
        
        if abs(py - py_calc) > 0.01:
            u_theta1 = -u_theta1
            
        theta1 = np.arctan(u_theta1) * 2
        
        # J4, J5, J6 require rotation matrix (target_quat)
        # For this simplified mobile platform test, we'll keep them neutral
        # and focus on the primary 3DOF positioning first.
        
        return [theta1, theta2, theta3, 0.0, 0.0, 0.0]

if __name__ == "__main__":
    ik = ZeroArmIK()
    # Test reaching a point
    angles = ik.solve_ik([0.2, 0.1, 0.3])
    print(f"IK Result: {angles}")
