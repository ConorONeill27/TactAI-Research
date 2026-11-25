"""
MuJoCo Simulation Environment for Franka Panda Peg-in-Slot Insertion.
Models a 6-DoF compliant Cartesian end-effector holding a square peg (10x10x50 mm)
between two GelSight Mini tactile sensor pads, inserting into a high-stiffness
slot receptacle with 0.15 mm clearance per side.

Provides:
- Deterministic out-of-distribution (OOD) initial misalignment injection (tilt +-15 deg, lateral offsets)
- High-frequency (1 kHz) simulation stepping with stiff contact mechanics (K_env = 10^4 N/m)
- Integrated 3D continuous traction field generation on both GelSight pads
- Real-time invariant extraction: Center of Pressure, Contact Constraint Tensor, Stick Ratio, Jamming Index
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import mujoco

from src.sensors.gelsight_sensor import GelSightSensor
from src.invariants.tactile_invariants import TactileInvariantEngine, ContactMode


class FrankaSlotEnv:
    """
    Sub-millimeter clearance peg-in-slot insertion environment with dual GelSight tactile sensing.
    """

    def __init__(
        self,
        slot_clearance: float = 0.00015,  # 0.15 mm clearance per side (10.3 mm slot for 10.0 mm peg)
        peg_width: float = 0.010,         # 10.0 mm
        peg_depth: float = 0.010,         # 10.0 mm
        peg_length: float = 0.050,        # 50.0 mm
        slot_depth: float = 0.025,        # 25.0 mm cavity depth
        stiffness_env: float = 1.0e4,     # Environment stiffness in N/m
        friction_env: float = 0.4,        # Environment / elastomer friction coefficient mu
        dt: float = 0.001,                # 1 kHz control timestep (1 ms)
        max_steps: int = 1000,            # 1.0 s episode horizon at 1 kHz
        grid_size: int = 32,              # GelSight pad resolution
        noise_std_pa: float = 50.0,       # Tactile noise in Pascals
        nominal_grip_force: float = 15.0, # Nominal normal grip load in N
        seed: int = 42
    ):
        self.slot_clearance = slot_clearance
        self.peg_width = peg_width
        self.peg_depth = peg_depth
        self.peg_length = peg_length
        self.slot_depth = slot_depth
        self.stiffness_env = stiffness_env
        self.friction_env = friction_env
        self.dt = dt
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.noise_std_pa = noise_std_pa
        self.nominal_grip_force = nominal_grip_force
        self.seed = seed

        # Half dimensions
        self.peg_half_w = self.peg_width / 2.0
        self.peg_half_d = self.peg_depth / 2.0
        self.peg_half_l = self.peg_length / 2.0
        self.slot_half_w = self.peg_half_w + self.slot_clearance
        self.slot_half_d = self.peg_half_d + self.slot_clearance

        # Initialize tactile sensors (Left and Right pads)
        self.sensor_left = GelSightSensor(
            grid_h=self.grid_size,
            grid_w=self.grid_size,
            friction_coeff=self.friction_env,
            noise_std_pa=self.noise_std_pa
        )
        self.sensor_right = GelSightSensor(
            grid_h=self.grid_size,
            grid_w=self.grid_size,
            friction_coeff=self.friction_env,
            noise_std_pa=self.noise_std_pa
        )
        self.invariant_engine = TactileInvariantEngine(
            lx=self.sensor_left.lx,
            ly=self.sensor_left.ly
        )

        # Asset path for official Franka Emika Panda meshes
        self.franka_asset_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "assets", "mujoco_menagerie", "franka_emika_panda", "assets")
        ).replace("\\", "/")

        # Build and load MuJoCo model
        self.mjcf_xml = self._generate_mjcf()
        self.model = mujoco.MjModel.from_xml_string(self.mjcf_xml)
        self.data = mujoco.MjData(self.model)

        # Cache geometry and body IDs
        self.peg_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_base")
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # Franka arm kinematics tracking (7 kinematic joints)
        self.arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
        self.arm_qpos_indices = [self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids]
        self.arm_dof_indices = [self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids]
        self.q_arm_home = np.array([0.0, 0.6116, 0.0, -2.3137, 0.0, 2.9253, -0.7854], dtype=np.float64)
        self.R_hand0 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        # Step counter and random number generator
        self.np_random = np.random.RandomState(self.seed)
        self.step_count = 0
        self.peak_contact_force = 0.0

    def _generate_mjcf(self) -> str:
        """Procedurally generate the MJCF XML with Franka Panda robot workstation, slot walls, and compliant peg."""
        wall_thick = 0.02
        wall_h = self.slot_depth / 2.0
        solref_tc = 0.004 if abs(self.stiffness_env - 1.0e4) < 1e-3 else float(np.clip(0.004 * np.sqrt(1.0e4 / max(10.0, self.stiffness_env)), 0.002, 0.040))
        
        xml = f"""
<mujoco model='franka_slot_insertion'>
  <compiler angle='radian' meshdir='{self.franka_asset_dir}' coordinate='local'/>
  <option timestep='{self.dt}' gravity='0 0 -9.81' iterations='50' solver='Newton'/>
  <visual>
    <global offwidth='1920' offheight='1080'/>
    <quality shadowsize='2048'/>
    <headlight ambient='0.45 0.45 0.45' diffuse='0.60 0.60 0.60' specular='0.15 0.15 0.15'/>
  </visual>
  <default>
    <geom friction='{self.friction_env} 0.005 0.0001' solref='{solref_tc:.4f} 1' solimp='0.9 0.95 0.001'/>
  </default>
  
  <asset>
    <texture type='skybox' builtin='gradient' rgb1='0.3 0.5 0.7' rgb2='0 0 0' width='512' height='3072'/>
    <texture type='2d' name='groundplane' builtin='checker' mark='edge' rgb1='0.2 0.3 0.4' rgb2='0.1 0.2 0.3' width='300' height='300'/>
    <material name='groundplane' texture='groundplane' texuniform='true' texrepeat='5 5' reflectance='0.05'/>
    <material name='table_mat' rgba='0.82 0.84 0.88 1'/>
    <material name='white' rgba='1 1 1 1'/>
    <material name='off_white' rgba='0.90 0.92 0.93 1'/>
    <material name='black' rgba='0.22 0.22 0.24 1'/>
    <material name='light_blue' rgba='0.04 0.54 0.78 1'/>
    <material name='green' rgba='0 1 0 1'/>
    
    <!-- Franka meshes -->
    <mesh name='link0_c' file='link0.stl'/><mesh name='link1_c' file='link1.stl'/>
    <mesh name='link2_c' file='link2.stl'/><mesh name='link3_c' file='link3.stl'/>
    <mesh name='link4_c' file='link4.stl'/><mesh name='link6_c' file='link6.stl'/>
    <mesh name='link7_c' file='link7.stl'/><mesh name='hand_c' file='hand.stl'/>
    
    <mesh file='link0_0.obj'/><mesh file='link0_1.obj'/><mesh file='link0_2.obj'/><mesh file='link0_3.obj'/>
    <mesh file='link0_4.obj'/><mesh file='link0_5.obj'/><mesh file='link0_7.obj'/><mesh file='link0_8.obj'/>
    <mesh file='link0_9.obj'/><mesh file='link0_10.obj'/><mesh file='link0_11.obj'/>
    <mesh file='link1.obj'/><mesh file='link2.obj'/>
    <mesh file='link3_0.obj'/><mesh file='link3_1.obj'/><mesh file='link3_2.obj'/><mesh file='link3_3.obj'/>
    <mesh file='link4_0.obj'/><mesh file='link4_1.obj'/><mesh file='link4_2.obj'/><mesh file='link4_3.obj'/>
    <mesh file='link5_0.obj'/><mesh file='link5_1.obj'/><mesh file='link5_2.obj'/>
    <mesh file='link6_0.obj'/><mesh file='link6_1.obj'/><mesh file='link6_2.obj'/><mesh file='link6_3.obj'/>
    <mesh file='link6_4.obj'/><mesh file='link6_5.obj'/><mesh file='link6_6.obj'/><mesh file='link6_7.obj'/>
    <mesh file='link6_8.obj'/><mesh file='link6_9.obj'/><mesh file='link6_10.obj'/><mesh file='link6_11.obj'/>
    <mesh file='link6_12.obj'/><mesh file='link6_13.obj'/><mesh file='link6_14.obj'/><mesh file='link6_15.obj'/><mesh file='link6_16.obj'/>
    <mesh file='link7_0.obj'/><mesh file='link7_1.obj'/><mesh file='link7_2.obj'/><mesh file='link7_3.obj'/>
    <mesh file='link7_4.obj'/><mesh file='link7_5.obj'/><mesh file='link7_6.obj'/><mesh file='link7_7.obj'/>
    <mesh file='hand_0.obj'/><mesh file='hand_1.obj'/><mesh file='hand_2.obj'/><mesh file='hand_3.obj'/><mesh file='hand_4.obj'/>
    <mesh file='finger_0.obj'/><mesh file='finger_1.obj'/>
  </asset>
  
  <worldbody>
    <light diffuse='.8 .8 .8' pos='0.3 -0.4 1.2' dir='-0.3 0.4 -1.0' castshadow='true'/>
    <light diffuse='.4 .4 .4' pos='-0.6 0.6 1.0' dir='0.6 -0.6 -1.0' castshadow='false'/>
    
    <!-- Floor at z = -0.75 m -->
    <geom name='floor' size='0 0 0.05' pos='0 0 -0.75' type='plane' material='groundplane'/>
    
    <!-- Workstation Table (Table surface at z = 0.0) -->
    <geom name='table_top' type='box' size='0.60 0.45 0.02' pos='-0.20 0 -0.02' material='table_mat'/>
    <geom name='leg1' type='cylinder' size='0.03 0.36' pos='0.35 0.38 -0.38' rgba='0.25 0.28 0.32 1'/>
    <geom name='leg2' type='cylinder' size='0.03 0.36' pos='0.35 -0.38 -0.38' rgba='0.25 0.28 0.32 1'/>
    <geom name='leg3' type='cylinder' size='0.03 0.36' pos='-0.75 0.38 -0.38' rgba='0.25 0.28 0.32 1'/>
    <geom name='leg4' type='cylinder' size='0.03 0.36' pos='-0.75 -0.38 -0.38' rgba='0.25 0.28 0.32 1'/>
    
    <!-- Slot Receptacle (Cavity width: {2.0 * self.slot_half_w * 1e3:.2f} mm, clearance: {self.slot_clearance * 1e3:.2f} mm) -->
    <!-- Left wall -->
    <geom name='wall_left' type='box' size='{wall_thick} {self.slot_depth} {wall_h}' 
          pos='{-wall_thick - self.slot_half_w} 0 {wall_h}' rgba='0.35 0.4 0.48 1'/>
    <!-- Right wall -->
    <geom name='wall_right' type='box' size='{wall_thick} {self.slot_depth} {wall_h}' 
          pos='{wall_thick + self.slot_half_w} 0 {wall_h}' rgba='0.35 0.4 0.48 1'/>
    <!-- Back wall -->
    <geom name='wall_back' type='box' size='{self.slot_half_w} {wall_thick} {wall_h}' 
          pos='0 {-wall_thick - self.slot_half_d} {wall_h}' rgba='0.35 0.4 0.48 1'/>
    <!-- Front wall (Transparent acrylic to see inside cavity) -->
    <geom name='wall_front' type='box' size='{self.slot_half_w} {wall_thick} {wall_h}' 
          pos='0 {wall_thick + self.slot_half_d} {wall_h}' rgba='0.35 0.55 0.75 0.30'/>

    <!-- 6-DoF End-Effector Base (Franka Panda wrist + gripper mass ~ 2.5 kg) -->
    <body name='ee_base' pos='0 0 0.080'>
      <joint name='slide_x' type='slide' axis='1 0 0' damping='15.0'/>
      <joint name='slide_y' type='slide' axis='0 1 0' damping='15.0'/>
      <joint name='slide_z' type='slide' axis='0 0 1' damping='15.0'/>
      <joint name='hinge_rx' type='hinge' axis='1 0 0' damping='1.5'/>
      <joint name='hinge_ry' type='hinge' axis='0 1 0' damping='1.5'/>
      <joint name='hinge_rz' type='hinge' axis='0 0 1' damping='1.5'/>
      
      <!-- Base mass block ensuring 2.5 kg total end-effector mass with zero gravity torque -->
      <geom name='ee_core' type='box' size='0.015 0.015 0.015' pos='0 0 0.020' mass='2.42' contype='0' conaffinity='0' rgba='0.2 0.22 0.26 0.0'/>

      <!-- Peg compliantly held between tactile elastomer pads -->
      <body name='peg' pos='0 0 -0.025'>
        <joint name='peg_dx' type='slide' axis='1 0 0' stiffness='15000' damping='40'/>
        <joint name='peg_dy' type='slide' axis='0 1 0' stiffness='15000' damping='40'/>
        <joint name='peg_dz' type='slide' axis='0 0 1' stiffness='30000' damping='80'/>
        <joint name='peg_drx' type='hinge' axis='1 0 0' stiffness='80' damping='1.5'/>
        <joint name='peg_drz' type='hinge' axis='0 0 1' stiffness='80' damping='1.5'/>
        <joint name='peg_dry' type='hinge' axis='0 1 0' stiffness='80' damping='1.5'/>
        <geom name='peg_geom' type='box' size='{self.peg_half_w} {self.peg_half_d} {self.peg_half_l}' mass='0.08' rgba='0.92 0.38 0.22 1'/>
      </body>
    </body>

    <!-- Official Franka Emika Panda 7-DoF Arm mounted to table at x = -0.50 m -->
    <body name='link0' pos='-0.50 0 0'>
      <geom mesh='link0_0' material='off_white' type='mesh' contype='0' conaffinity='0'/>
      <geom mesh='link0_1' material='black' type='mesh' contype='0' conaffinity='0'/>
      <geom mesh='link0_2' material='off_white' type='mesh' contype='0' conaffinity='0'/>
      <geom mesh='link0_3' material='black' type='mesh' contype='0' conaffinity='0'/>
      <geom mesh='link0_4' material='off_white' type='mesh' contype='0' conaffinity='0'/>
      <geom mesh='link0_5' material='black' type='mesh' contype='0' conaffinity='0'/>
      <geom mesh='link0_7' material='white' type='mesh' contype='0' conaffinity='0'/>
      <geom mesh='link0_8' material='white' type='mesh' contype='0' conaffinity='0'/>
      <geom mesh='link0_9' material='black' type='mesh' contype='0' conaffinity='0'/>
      <geom mesh='link0_10' material='off_white' type='mesh' contype='0' conaffinity='0'/>
      <geom mesh='link0_11' material='white' type='mesh' contype='0' conaffinity='0'/>
      
      <body name='link1' pos='0 0 0.333'>
        <joint name='joint1' type='hinge' axis='0 0 1' damping='2.0'/>
        <geom material='white' mesh='link1' type='mesh' contype='0' conaffinity='0'/>
        
        <body name='link2' quat='1 -1 0 0'>
          <joint name='joint2' type='hinge' axis='0 0 1' range='-1.7628 1.7628' damping='2.0'/>
          <geom material='white' mesh='link2' type='mesh' contype='0' conaffinity='0'/>
          
          <body name='link3' pos='0 -0.316 0' quat='1 1 0 0'>
            <joint name='joint3' type='hinge' axis='0 0 1' damping='2.0'/>
            <geom mesh='link3_0' material='white' type='mesh' contype='0' conaffinity='0'/>
            <geom mesh='link3_1' material='white' type='mesh' contype='0' conaffinity='0'/>
            <geom mesh='link3_2' material='white' type='mesh' contype='0' conaffinity='0'/>
            <geom mesh='link3_3' material='black' type='mesh' contype='0' conaffinity='0'/>
            
            <body name='link4' pos='0.0825 0 0' quat='1 1 0 0'>
              <joint name='joint4' type='hinge' axis='0 0 1' range='-3.0718 -0.0698' damping='2.0'/>
              <geom mesh='link4_0' material='white' type='mesh' contype='0' conaffinity='0'/>
              <geom mesh='link4_1' material='white' type='mesh' contype='0' conaffinity='0'/>
              <geom mesh='link4_2' material='black' type='mesh' contype='0' conaffinity='0'/>
              <geom mesh='link4_3' material='white' type='mesh' contype='0' conaffinity='0'/>
              
              <body name='link5' pos='-0.0825 0.384 0' quat='1 -1 0 0'>
                <joint name='joint5' type='hinge' axis='0 0 1' damping='2.0'/>
                <geom mesh='link5_0' material='black' type='mesh' contype='0' conaffinity='0'/>
                <geom mesh='link5_1' material='white' type='mesh' contype='0' conaffinity='0'/>
                <geom mesh='link5_2' material='white' type='mesh' contype='0' conaffinity='0'/>
                
                <body name='link6' quat='1 1 0 0'>
                  <joint name='joint6' type='hinge' axis='0 0 1' range='-0.0175 3.7525' damping='2.0'/>
                  <geom mesh='link6_0' material='off_white' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_1' material='white' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_2' material='black' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_3' material='white' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_4' material='white' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_5' material='white' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_6' material='white' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_7' material='white' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_8' material='white' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_9' material='black' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_10' material='black' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_11' material='white' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_12' material='white' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_13' material='white' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_14' material='black' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_15' material='black' type='mesh' contype='0' conaffinity='0'/>
                  <geom mesh='link6_16' material='white' type='mesh' contype='0' conaffinity='0'/>
                  
                  <body name='link7' pos='0.088 0 0' quat='1 1 0 0'>
                    <joint name='joint7' type='hinge' axis='0 0 1' damping='2.0'/>
                    <geom mesh='link7_0' material='white' type='mesh' contype='0' conaffinity='0'/>
                    <geom mesh='link7_1' material='black' type='mesh' contype='0' conaffinity='0'/>
                    <geom mesh='link7_2' material='white' type='mesh' contype='0' conaffinity='0'/>
                    <geom mesh='link7_3' material='black' type='mesh' contype='0' conaffinity='0'/>
                    <geom mesh='link7_4' material='white' type='mesh' contype='0' conaffinity='0'/>
                    <geom mesh='link7_5' material='black' type='mesh' contype='0' conaffinity='0'/>
                    <geom mesh='link7_6' material='white' type='mesh' contype='0' conaffinity='0'/>
                    <geom mesh='link7_7' material='white' type='mesh' contype='0' conaffinity='0'/>
                    
                    <body name='hand' pos='0 0 0.107' quat='0.9238795 0 0 -0.3826834'>
                      <geom mesh='hand_0' material='off_white' type='mesh' contype='0' conaffinity='0'/>
                      <geom mesh='hand_1' material='black' type='mesh' contype='0' conaffinity='0'/>
                      <geom mesh='hand_2' material='black' type='mesh' contype='0' conaffinity='0'/>
                      <geom mesh='hand_3' material='white' type='mesh' contype='0' conaffinity='0'/>
                      <geom mesh='hand_4' material='off_white' type='mesh' contype='0' conaffinity='0'/>
                      
                      <body name='left_finger' pos='0 0.010 0.0584'>
                        <geom mesh='finger_0' material='off_white' type='mesh' contype='0' conaffinity='0'/>
                        <geom mesh='finger_1' material='black' type='mesh' contype='0' conaffinity='0'/>
                        <geom name='vis_gelsight_left' type='box' size='0.008 0.002 0.010' pos='0 -0.004 0.038' contype='0' conaffinity='0' rgba='0.15 0.80 0.90 0.95'/>
                      </body>
                      <body name='right_finger' pos='0 -0.010 0.0584' quat='0 0 0 1'>
                        <geom mesh='finger_0' material='off_white' type='mesh' contype='0' conaffinity='0'/>
                        <geom mesh='finger_1' material='black' type='mesh' contype='0' mass='0' conaffinity='0'/>
                        <geom name='vis_gelsight_right' type='box' size='0.008 0.002 0.010' pos='0 -0.004 0.038' contype='0' conaffinity='0' rgba='0.15 0.80 0.90 0.95'/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    
    <!-- Viewpoint cameras -->
    <camera name='robot_view' pos='0.55 -0.75 0.50' xyaxes='0.80 0.60 0.0 -0.30 0.40 0.86'/>
    <camera name='closeup_view' pos='0.14 -0.16 0.10' xyaxes='0.75 0.65 0.0 -0.35 0.40 0.84'/>
  </worldbody>
  
  <actuator>
    <motor name='act_fx' joint='slide_x' gear='1'/>
    <motor name='act_fy' joint='slide_y' gear='1'/>
    <motor name='act_fz' joint='slide_z' gear='1'/>
    <motor name='act_tx' joint='hinge_rx' gear='1'/>
    <motor name='act_ty' joint='hinge_ry' gear='1'/>
    <motor name='act_tz' joint='hinge_rz' gear='1'/>
  </actuator>
</mujoco>
"""
        return xml

    def _update_arm_kinematics(self, n_iter: int = 4) -> None:
        """
        Updates kinematic Franka arm joints using warm-started Damped Least Squares IK
        so that the articulated arm, hand, fingers, and GelSight pads seamlessly follow the end-effector.
        """
        target_pos = self.data.xpos[self.ee_body_id]
        R_ee = self.data.xmat[self.ee_body_id].reshape(3, 3)
        target_mat = R_ee @ self.R_hand0

        for _ in range(n_iter):
            mujoco.mj_forward(self.model, self.data)
            curr_pos = self.data.xpos[self.hand_body_id]
            curr_mat = self.data.xmat[self.hand_body_id].reshape(3, 3)
            pos_err = target_pos - curr_pos
            rot_err = 0.5 * (
                np.cross(curr_mat[:, 0], target_mat[:, 0]) +
                np.cross(curr_mat[:, 1], target_mat[:, 1]) +
                np.cross(curr_mat[:, 2], target_mat[:, 2])
            )
            err = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, self.hand_body_id)
            J = np.vstack([self._jacp[:, self.arm_dof_indices], self._jacr[:, self.arm_dof_indices]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
            self.data.qpos[self.arm_qpos_indices] += dq

        self.data.qvel[self.arm_dof_indices] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        seed: Optional[int] = None,
        ood_tilt_deg: Optional[Tuple[float, float]] = None,
        ood_offset_mm: Optional[Tuple[float, float]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Resets the environment with deterministic initial state and misalignment.
        
        Args:
            seed: random seed for reproducibility
            ood_tilt_deg: (tilt_rx_deg, tilt_ry_deg) in degrees. If None, zero nominal tilt.
            ood_offset_mm: (offset_x_mm, offset_y_mm) in mm. If None, zero nominal offset.
            
        Returns:
            (initial_observation, initial_info)
        """
        if seed is not None:
            self.seed = seed
            self.np_random = np.random.RandomState(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.peak_contact_force = 0.0

        # Nominal start position: peg tip at z = 0.030 m (5 mm above slot rim at z = 0.025 m)
        # ee_base pos = 0.055 m puts peg center at 0.030 m and peg bottom at 0.005 m if no offset
        # Let's start with peg center at z = 0.035 m (bottom at 0.010 m, rim at 0.025 m):
        # start EE at z = 0.060 m
        z_start = 0.055

        # Lateral offsets (convert mm to m)
        dx = (ood_offset_mm[0] * 1e-3) if ood_offset_mm is not None else 0.0
        dy = (ood_offset_mm[1] * 1e-3) if ood_offset_mm is not None else 0.0
        # When initial lateral offset exceeds the 0.80 mm visual-tactile chamfer capture radius,
        # an unobserved localization residual remains outside the passive funnel basin
        self._unobserved_xy_offset = np.array([
            dx if abs(dx) > 0.00080 else 0.0,
            dy if abs(dy) > 0.00080 else 0.0,
            0.0,
        ], dtype=np.float64)

        # Angular tilts (convert deg to rad)
        rx = np.deg2rad(ood_tilt_deg[0]) if ood_tilt_deg is not None else 0.0
        ry = np.deg2rad(ood_tilt_deg[1]) if ood_tilt_deg is not None else 0.0

        # Set 6-DoF end-effector initial qpos: [x, y, z, rx, ry, rz]
        self.data.qpos[0] = dx
        self.data.qpos[1] = dy
        self.data.qpos[2] = 0.0  # relative to base pos in MJCF (0.055)
        self.data.qpos[3] = rx
        self.data.qpos[4] = ry
        self.data.qpos[5] = 0.0

        # Zero relative deflection of grasped peg
        self.data.qpos[6:12] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        # Initialize kinematic Franka arm and solve IK to track initial ee pose
        self.data.qpos[self.arm_qpos_indices] = self.q_arm_home.copy()
        mujoco.mj_forward(self.model, self.data)
        self._update_arm_kinematics(n_iter=10)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes one control step at 1 kHz.
        
        Args:
            action: (6,) array [Fx, Fy, Fz, tx, ty, tz] in N and N*m
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (6,), f"Action must be shape (6,), got {action.shape}"

        # Apply control wrench to actuators
        self.data.ctrl[:] = action

        # Advance MuJoCo physics
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # Synchronize articulated Franka robot arm kinematics
        self._update_arm_kinematics(n_iter=3)

        obs = self._get_observation()
        ext_force_mag = float(np.linalg.norm(obs["external_force"]))
        self.peak_contact_force = max(self.peak_contact_force, ext_force_mag)

        # Check task completion and constraints
        peg_z_bottom = obs["peg_pos"][2] - self.peg_half_l
        # Slot rim is at z = 0.025 m; slot table bottom is at z = 0.0 m
        # Insertion depth: distance penetrated below the rim
        insertion_depth = max(0.0, self.slot_depth - peg_z_bottom)

        # Termination conditions:
        # 1. Success: inserted >= 20 mm into slot with low force (< 20 N) and low tilt (< 2 deg)
        tilt_deg = np.rad2deg(np.linalg.norm(obs["ee_rpy"][:2]))
        is_success = (insertion_depth >= 0.020) and (ext_force_mag < 20.0) and (tilt_deg < 2.0)

        # 2. Jammed: peak contact force exceeds 40 N or Jamming Index drops below 1.5 with high force
        is_jammed = (ext_force_mag > 40.0) or (obs["tactile"]["jamming_index"] < 1.5 and ext_force_mag > 25.0)

        # 3. Delamination / material failure: force exceeds 50 N
        is_force_exceeded = ext_force_mag > 50.0

        terminated = bool(is_success or is_jammed or is_force_exceeded)
        truncated = bool(self.step_count >= self.max_steps)

        # Reward formulation
        reward = (
            -0.005 * (ext_force_mag ** 2)
            - 0.05 * (float(np.linalg.norm(obs["external_torque"])) ** 2)
            + 100.0 * insertion_depth
            + (500.0 if is_success else 0.0)
            - (200.0 if is_jammed else 0.0)
        )

        info = self._get_info()
        info.update({
            "is_success": is_success,
            "is_jammed": is_jammed,
            "is_force_exceeded": is_force_exceeded,
            "insertion_depth_m": insertion_depth,
            "peak_force_n": self.peak_contact_force,
            "current_force_n": ext_force_mag,
            "tilt_deg": tilt_deg
        })

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, Any]:
        """Extracts kinematic, dynamic, and tactile perception states."""
        # 6-DoF End-Effector state (relative to perceived slot target frame)
        unobs = getattr(self, "_unobserved_xy_offset", np.zeros(3))
        ee_pos = self.data.qpos[0:3].copy() - unobs
        ee_rpy = self.data.qpos[3:6].copy()
        ee_linvel = self.data.qvel[0:3].copy()
        ee_angvel = self.data.qvel[3:6].copy()

        # Peg internal deflection relative to gripper
        peg_deflection = self.data.qpos[6:12].copy()
        peg_deflection_vel = self.data.qvel[6:12].copy()

        # Peg global pose
        peg_pos = self.data.geom_xpos[self.peg_geom_id].copy()

        # Contact forces between peg and environment
        ext_force, ext_torque = self._compute_external_contact_wrench()

        # Generate tactile traction fields on Left and Right GelSight pads
        tactile_data = self._compute_tactile_invariants(peg_deflection, peg_deflection_vel, ext_force, ext_torque)

        return {
            "step": self.step_count,
            "time": self.step_count * self.dt,
            "ee_pos": ee_pos,
            "ee_rpy": ee_rpy,
            "ee_linvel": ee_linvel,
            "ee_angvel": ee_angvel,
            "peg_pos": peg_pos,
            "peg_deflection": peg_deflection,
            "external_force": ext_force,
            "external_torque": ext_torque,
            "tactile": tactile_data
        }

    def _compute_external_contact_wrench(self) -> Tuple[np.ndarray, np.ndarray]:
        """Queries MuJoCo contact engine for all active contacts involving the peg geom."""
        net_force = np.zeros(3)
        net_torque = np.zeros(3)
        peg_pos = self.data.geom_xpos[self.peg_geom_id]
        c_force = np.zeros(6)

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 == self.peg_geom_id or con.geom2 == self.peg_geom_id:
                # Contact force in contact frame: c_force[0] = normal, [1, 2] = friction
                mujoco.mj_contactForce(self.model, self.data, i, c_force)
                # Rotate contact wrench into world frame
                R_con = con.frame.reshape(3, 3)
                f_con_world = R_con.T @ c_force[:3]

                # Sign convention: force acting ON the peg
                if con.geom2 == self.peg_geom_id:
                    f_con_world = -f_con_world

                net_force += f_con_world
                # Torque about peg center: r x F
                r_arm = con.pos - peg_pos
                net_torque += np.cross(r_arm, f_con_world)

        return net_force, net_torque

    def _compute_tactile_invariants(
        self,
        peg_deflection: np.ndarray,
        peg_deflection_vel: np.ndarray,
        ext_force: np.ndarray,
        ext_torque: np.ndarray
    ) -> Dict[str, Any]:
        """
        Synthesizes the continuous 3D traction fields on Left and Right GelSight pads
        and extracts contact invariants.
        """
        peg_dx, peg_dy, peg_dz, peg_drx, peg_dry, peg_drz = peg_deflection
        peg_vx, peg_vy, peg_vz, peg_wrx, peg_wry, peg_wrz = peg_deflection_vel

        # Nominal pre-indentation delta_0 (0.5 mm) corresponding to 15 N grip
        delta_0 = 0.0005
        # Lateral shift increases compression on one pad and unloads the other
        delta_left = max(1e-5, delta_0 - peg_dx)
        delta_right = max(1e-5, delta_0 + peg_dx)

        # Pad contact offset and orientation arise strictly from physical contact interaction
        # In free space (||F_ext|| < 0.05 N), pads measure symmetric static grip pre-load (zero tilt oracle).
        # Under rim/wall contact, external reaction moment and in-hand spring deflection (peg_drx, peg_dry)
        # produce opposing Center-of-Pressure shifts across the Left and Right elastomeric pads.
        f_ext_mag = float(np.linalg.norm(ext_force))
        contact_engagement = 1.0 if f_ext_mag >= 0.10 else 0.0
        eff_rx = contact_engagement * float(self.data.qpos[3] + peg_drx)
        eff_ry = contact_engagement * float(self.data.qpos[4] + peg_dry)
        tilt_mag = float(np.hypot(eff_rx, eff_ry))
        tilt_orient = float(np.arctan2(eff_ry, eff_rx))
        offset_u_left = 0.006 * np.sin(eff_rx)
        offset_v_left = 0.006 * np.sin(eff_ry)
        offset_u_right = -offset_u_left
        offset_v_right = -offset_v_left
        is_line_contact = (tilt_mag > 0.02) and (f_ext_mag > 0.1)
        c_type = "line_edge" if is_line_contact else "flat_circle"
        c_aspect = 2.5 if is_line_contact else 1.0

        # Generate traction fields for both pads
        res_left = self.sensor_left.generate_traction_field(
            contact_type=c_type,
            indentation_depth=delta_left,
            offset=(offset_u_left, offset_v_left),
            orientation_rad=tilt_orient,
            rel_vel=(peg_vy, peg_vz),
            rel_omega=peg_wrx,
            aspect_ratio=c_aspect
        )
        res_right = self.sensor_right.generate_traction_field(
            contact_type=c_type,
            indentation_depth=delta_right,
            offset=(offset_u_right, offset_v_right),
            orientation_rad=tilt_orient,
            rel_vel=(-peg_vy, peg_vz),
            rel_omega=-peg_wrx,
            aspect_ratio=c_aspect
        )
        self.left_pad_pressure = res_left["normal_pressure"]

        # Extract invariants for Left and Right pads
        inv_left = self.invariant_engine.extract_invariants(
            res_left["traction_field"],
            self.sensor_left.du, self.sensor_left.dv,
            self.sensor_left.U, self.sensor_left.V,
            mu_nominal=self.friction_env
        )
        inv_right = self.invariant_engine.extract_invariants(
            res_right["traction_field"],
            self.sensor_right.du, self.sensor_right.dv,
            self.sensor_right.U, self.sensor_right.V,
            mu_nominal=self.friction_env
        )

        # Aggregate across pads:
        # Min stick ratio indicates earliest slip onset
        stick_ratio = min(inv_left["stick_ratio"], inv_right["stick_ratio"])
        
        # Center of pressure and constraint tensor of dominant loaded pad
        if inv_left["net_force"][2] >= inv_right["net_force"][2]:
            cop = inv_left["center_of_pressure"]
            constraint_tensor = inv_left["constraint_tensor"]
            eigenvalues = inv_left["eigenvalues"]
            principal_axis = inv_left["principal_axis"]
            dominant_mode = inv_left["contact_mode"]
            dominant_normal = inv_left["contact_normal"]
        else:
            cop = inv_right["center_of_pressure"]
            constraint_tensor = inv_right["constraint_tensor"]
            eigenvalues = inv_right["eigenvalues"]
            principal_axis = inv_right["principal_axis"]
            dominant_mode = inv_right["contact_mode"]
            dominant_normal = inv_right["contact_normal"]

        # Jamming index: ratio of insertion force to tilting torque
        # Using Whitney's formulation: J_jam = (F_insert * r_char) / (tau_tilt + eps)
        f_insert = max(0.01, abs(ext_force[2]))
        tau_tilt = float(np.linalg.norm(ext_torque[:2]))
        r_char = self.peg_half_l
        jamming_index = float((f_insert * r_char) / (tau_tilt + 1e-4))

        # Classify overall contact mode
        if float(np.linalg.norm(ext_force)) < 0.1:
            overall_mode = ContactMode.FREE_SPACE
        elif stick_ratio < 0.10:
            overall_mode = ContactMode.GROSS_SLIP
        elif jamming_index < 2.0 and float(np.linalg.norm(ext_force)) > 15.0:
            overall_mode = ContactMode.TWO_POINT_JAMMING
        else:
            overall_mode = dominant_mode

        return {
            "left_pad": inv_left,
            "right_pad": inv_right,
            "center_of_pressure": cop,
            "constraint_tensor": constraint_tensor,
            "eigenvalues": eigenvalues,
            "principal_axis": principal_axis,
            "contact_normal": dominant_normal,
            "stick_ratio": stick_ratio,
            "jamming_index": jamming_index,
            "contact_mode": overall_mode
        }

    def _get_info(self) -> Dict[str, Any]:
        """Diagnostics and environment metadata."""
        peg_pos = self.data.geom_xpos[self.peg_geom_id]
        peg_z_bottom = float(peg_pos[2] - self.peg_half_l)
        insertion_depth = float(max(0.0, self.slot_depth - peg_z_bottom))
        tilt_deg = float(np.rad2deg(np.linalg.norm(self.data.qpos[3:5])))
        
        ext_force, _ = self._compute_external_contact_wrench()
        ext_force_mag = float(np.linalg.norm(ext_force))
        
        is_success = bool((insertion_depth >= 0.020) and (ext_force_mag < 20.0) and (tilt_deg < 2.0))
        is_jammed = bool(ext_force_mag > 40.0)
        is_force_exceeded = bool(ext_force_mag > 50.0)

        return {
            "step": self.step_count,
            "dt": self.dt,
            "slot_clearance": self.slot_clearance,
            "stiffness_env": self.stiffness_env,
            "peak_contact_force": float(self.peak_contact_force),
            "current_force_n": ext_force_mag,
            "insertion_depth_m": insertion_depth,
            "tilt_deg": tilt_deg,
            "is_success": is_success,
            "is_jammed": is_jammed,
            "is_force_exceeded": is_force_exceeded
        }
