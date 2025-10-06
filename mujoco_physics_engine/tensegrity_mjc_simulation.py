import multiprocessing
from pathlib import Path
from typing import List
from PIL import Image

import mujoco
import numpy as np
import scipy as sp

from mujoco_physics_engine.cable_motor import DCMotor
from mujoco_physics_engine.mujoco_simulation import AbstractMuJoCoSimulator
from mujoco_physics_engine.pid import PID

def debug_print(message, filename="tensegrity_mjc_simulation.py", debug_enabled=False):
    """Print debug messages with filename prefix if debug is enabled"""
    if debug_enabled:
        print(f"DEBUG {filename}: {message}")


class TensegrityMuJoCoSimulator(AbstractMuJoCoSimulator):
    """
    MuJoCo Simulator class for two joined tensegrities.
    """
    def __init__(self,
                 xml_path: Path,
                 visualize: bool = True,
                 render_size: (int, int) = (720, 720),
                 render_fps: int = 20,
                 num_actuated_cables: int = 12,
                 num_rods: int = 3,
                 obs_dim: int | None = None,
                 obs_mode: str = "tier2",
                 debug_enabled: bool = False,
                 controller_kp: float = 2.0,
                 controller_ki: float = 0.0,
                 controller_kd: float = 1.0):
        super().__init__(xml_path, visualize, render_size, render_fps)
        self.debug_enabled = debug_enabled
        self.min_cable_length = 0.6 # unit: meters*10 # NOTE: was 0.6 but changed to match PID default
        self.max_cable_length = 1.6 # unit: meters*10 # NOTE: was 2.4 but changed to match PID default
        self.n_actuators = num_actuated_cables
        self.curr_ctrl = [0.0 for _ in range(num_actuated_cables)]
        self.pids = [PID(Kp=controller_kp, Ki=controller_ki, Kd=controller_kd, dt=self.dt, debug_enabled=self.debug_enabled) for _ in range(num_actuated_cables)] # Pass correct timestep
        print(self.pids[0].Kp, self.pids[0].Ki, self.pids[0].Kd)
        self.cable_motors = [DCMotor(debug_enabled=self.debug_enabled) for _ in range(num_actuated_cables)]
        self.n_rods = num_rods
        self.n_cables = self.mjc_model.tendon_stiffness.shape[0]
        # NOTE: start here for debugging the zero inputs 09/30/2025
        # self.actuated_ids = (list(range(num_actuated_cables // 2))
        #                      + list(range(self.n_cables // 2, self.n_cables // 2 + num_actuated_cables // 2)))
        self.actuated_ids = [0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20]  # All vertex-to-bar cables
        debug_print(f"actuated_ids: {self.actuated_ids}", "tensegrity_mjc_simulation.py", self.debug_enabled)
        # Observation mode: 'tier2' (96D) or 'legacy104' (104D)
        self.obs_mode = obs_mode if obs_mode in ("tier2", "legacy104") else "tier2"
        self.obs_dim = (96 if self.obs_mode == "tier2" else 104) if obs_dim is None else obs_dim

        # Tuple of cable end point of attachment sites' names
        self.cable_sites = [
            # robot 1
            ("t1_s_3_b5", "t1_s_b5_3"),
            ("t1_s_1_b3", "t1_s_b3_1"),
            ("t1_s_5_b1", "t1_s_b1_5"),
            ("t1_s_0_b2", "t1_s_b2_0"),
            ("t1_s_4_b0", "t1_s_b0_4"),
            ("t1_s_2_b4", "t1_s_b4_2"),
            ("t1_s_3_5", "t1_s_5_3"),
            ("t1_s_1_3", "t1_s_3_1"),
            ("t1_s_1_5", "t1_s_5_1"),
            ("t1_s_0_2", "t1_s_2_0"),
            ("t1_s_0_4", "t1_s_4_0"),
            ("t1_s_2_4", "t1_s_4_2"),
            ("t1_s_2_5", "t1_s_5_2"),
            ("t1_s_0_3", "t1_s_3_0"),
            ("t1_s_1_4", "t1_s_4_1"),

            # robot 2
            ("t2_s_3_b5", "t2_s_b5_3"),
            ("t2_s_1_b3", "t2_s_b3_1"),
            ("t2_s_5_b1", "t2_s_b1_5"),
            ("t2_s_0_b2", "t2_s_b2_0"),
            ("t2_s_4_b0", "t2_s_b0_4"),
            ("t2_s_2_b4", "t2_s_b4_2"),
            ("t2_s_3_5", "t2_s_5_3"),
            ("t2_s_1_3", "t2_s_3_1"),
            ("t2_s_1_5", "t2_s_5_1"),
            ("t2_s_0_2", "t2_s_2_0"),
            ("t2_s_0_4", "t2_s_4_0"),
            ("t2_s_2_4", "t2_s_4_2"),
            ("t2_s_2_5", "t2_s_5_2"),
            ("t2_s_0_3", "t2_s_3_0"),
            ("t2_s_1_4", "t2_s_4_1")
        ]

        # List of end-cap sites (names)
        self.end_pts = [
            "t1_s0", "t1_s1", "t1_s2", "t1_s3", "t1_s4", "t1_s5",
            "t2_s0", "t2_s1", "t2_s2", "t2_s3", "t2_s4", "t2_s5",
        ]
        self.stiffness = self.mjc_model.tendon_stiffness.copy()  # Copy of original cable stiffnesses

        # --- Buffers and mappings for Tier-2 observation ---
        self.prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        self.prev_cable_lengths = np.zeros(self.n_actuators, dtype=np.float32)
        self.prev_COM_pos = None
        # strain sensors: assume first 18 tendons by default
        self.strain_tendon_ids = list(range(min(18, self.n_cables)))
        # IMU (rods) geom names present in XML sensors
        self.imu_geom_names = ["t1_r01", "t1_r23", "t1_r45", "t2_r01", "t2_r23", "t2_r45"]

    def bring_to_grnd(self):
        """
        Finds the z-translation that would bring the lowest end cap to the ground, and aplies it to the robot
        """
        self.forward()
        qpos = self.mjc_data.qpos.copy().reshape(-1, 7)
        end_pts = self.get_endpts().reshape(-1, 3)
        min_z = end_pts[:, 2].min()
        reset_ground_height = 0.175 # final height of end cap above ground
        qpos[:, 2] -= min_z - reset_ground_height
        self.mjc_data.qpos = qpos.reshape(1, -1)

    def reset(self):
        """
        Resets the robots as if it was just instantiated from the xml file.
        """
        super().reset()
        self.bring_to_grnd()
        # Store the original state
        self.prev_pos = None
        self.step_count = 0

        for motor in self.cable_motors:
            motor.reset_omega_t()
        
        # Initialize Tier-2 buffers
        self.prev_action[:] = 0.0
        self.prev_cable_lengths = self._get_actuated_cable_lengths()
        self.prev_COM_pos = self._compute_COM_position()
        self.prev_imu_grav = self._get_IMU_gravity_vectors()

        # Return observation for RL using selected mode
        observation = self.get_observation()
        return observation

    def reset_actuators(self):
        for motor in self.cable_motors:
            motor.reset_omega_t()

        for pid in self.pids:
            pid.reset()


    def sim_step(self, target_lengths=None):
        """
        Takes a single simulation step given target_lengths from RL policy.
        """
        ctrl_idx = 0

        # NOTE: double check that target_lengths is in [0, 1] range

        # Adding some debugging here (Setph)

        # self.apply_random_perturbation() #can comment

        # Convert target_lengths to controls in [-1, 1]
        if target_lengths is not None:
            controls = np.zeros(self.n_actuators)  # NumPy array
            for i in range(len(target_lengths)):
                lengths = target_lengths[i]
                rest_length = self.mjc_model.tendon_lengthspring[self.actuated_ids[i], 0]

                # # Normalize lengths to [0, 1]
                # norm_length = (lengths - self.min_cable_length) / (self.max_cable_length - self.min_cable_length)
                # norm_length = np.clip(norm_length, 0.0, 1.0)

                # Compute control signal using PID
                s0 = self.mjc_data.sensor(f"pos_{self.cable_sites[self.actuated_ids[i]][0]}").data
                s1 = self.mjc_data.sensor(f"pos_{self.cable_sites[self.actuated_ids[i]][1]}").data
                debug_print(f"actuated_ids[{i}]: {self.actuated_ids[i]}", "tensegrity_mjc_simulation.py", self.debug_enabled)
                debug_print(f"cable_sites[{self.actuated_ids[i]}]: {self.cable_sites[self.actuated_ids[i]]}", "tensegrity_mjc_simulation.py", self.debug_enabled)
                debug_print(f"cable_sites[{self.actuated_ids[i]}][0]: {self.cable_sites[self.actuated_ids[i]][0]}", "tensegrity_mjc_simulation.py", self.debug_enabled)
                curr_length = np.linalg.norm(s1 - s0)
                
                ctrl, _ = self.pids[i].update_control_by_target_norm_length(curr_length, lengths, rest_length, self.min_cable_length, self.max_cable_length)
                controls[i] = -1.0*ctrl
                # print(f"PID Control for cable {i} (actuated_id {self.actuated_ids[i]}): {ctrl}, Target norm length: {lengths}, Current length: {curr_length}, Rest length: {rest_length}")

            # Constrain cables that are mirrored to have same control as their pair
            for i in range(3):
                controls[i+6] = controls[i+3]
                # print(f"Control for cable {i+6} (actuated_id {self.actuated_ids[i+6]}): {controls[i+6]} (mirrored from cable {i+3})")

        # self.forward()
        # for i in range(len(self.cable_sites)):
        #     # ... existing code ...
        #     if controls is not None and i in self.actuated_ids:
        #         # Find the position in the action vector
        #         action_idx = list(self.actuated_ids).index(i)
        #         ctrl = np.array(controls[action_idx])
                
        #         # Compute change in cable rest lengths
        #         dl = self.cable_motors[action_idx].compute_cable_length_delta(ctrl, self.dt)
        #         rest_length = rest_length - dl
        #         self.mjc_model.tendon_lengthspring[i] = rest_length

        # self.forward()
        for i in range(len(self.cable_sites)):
            if controls is not None and i in self.actuated_ids:
                # Find the position in the action vector
                action_idx = list(self.actuated_ids).index(i)
                ctrl = np.array(controls[action_idx])
                
                # Get the specific rest length for THIS cable
                cable_rest_length = self.mjc_model.tendon_lengthspring[i, 0]
                
                # Compute change in cable rest lengths
                dl = self.cable_motors[action_idx].compute_cable_length_delta(ctrl, self.dt)
                
                # Update rest length for THIS cable only
                new_rest_length = cable_rest_length + dl  # Changed from - to +
                
                # Optional: Add bounds to prevent extreme contractions
                new_rest_length = np.clip(new_rest_length, self.min_cable_length, self.max_cable_length)
                
                # Apply the new rest length
                self.mjc_model.tendon_lengthspring[i] = new_rest_length
                # print(f"Cable {i} (actuated_id {self.actuated_ids[action_idx]}): Rest length updated from {cable_rest_length} to {new_rest_length} using dl={dl} and ctrl={ctrl}")
                
                # Debug (fixed for numpy arrays)
                if isinstance(ctrl, np.ndarray):
                    ctrl_val = ctrl.item() if ctrl.size == 1 else ctrl[0]
                else:
                    ctrl_val = ctrl
                    
                if isinstance(dl, np.ndarray):
                    dl_val = dl.item() if dl.size == 1 else dl[0]
                else:
                    dl_val = dl

                debug_print(f"DL: {dl_val}, Control: {ctrl_val}", "tensegrity_mjc_simulation.py", self.debug_enabled)

        mujoco.mj_step(self.mjc_model, self.mjc_data)
        self.forward()

        # Get end points for locomotion reward
        end_pts = self.get_endpts()
        robot_pos = end_pts.mean(axis=0)  # Use the mean of end points as the robot's position
        
        # Calculate forward velocity reward
        velocity_reward = 0.0
        # print("Robot position: ", self.prev_pos)
        if hasattr(self, 'prev_pos') and self.prev_pos is not None:
            # Calculate XY plane displacement
            xy_displacement = robot_pos[:2] - self.prev_pos[:2]  # [x, y] only
            xy_speed = np.linalg.norm(xy_displacement) / self.dt
            
            velocity_reward = xy_speed # Reward any XY movement with large magnitude

        if hasattr(self, 'prev_pos') and self.prev_pos is not None:
            # Calculate XY plane displacement
            xy_displacement = robot_pos[:2] - self.prev_pos[:2]  # [x, y] only
            # Reward positive forward velocity (assuming +x is forward direction)
            forward_velocity = xy_displacement[0] / self.dt  # x-component of velocity
            if forward_velocity > 0:
                velocity_reward += forward_velocity # Additional reward for forward movement
        
        # Add distance-based reward (total distance from origin)
        distance_reward = self.calculate_omnidirectional_distance_reward(robot_pos)

        penalties = self.calculate_anti_exploit_penalties(robot_pos, controls) #0


        # Notes with Jue 10/03/2025:
        # Use IMU speed and maybe acceleration readings
        # reward only axis of cylinder rotation. 
        # penalize rotation of other two
        # put the IMU readings in the observation
        # try to plot average endpoint x,y,z over time
        # look at weighting
        # plot how the reward evolves during training, hopefully stored in the zip file.

        # Old IMU Reward Term
        # # Reward changes in IMU orientation
        # # Reward changes in IMU orientation - encourage rotation in one direction
        # imu_reward = 0.0
        # if hasattr(self, 'prev_imu_grav'):
        #     current_imu_grav = self._get_IMU_gravity_vectors()
        #     # Calculate change in orientation (gravity vector change)
        #     # Reshape to get individual IMU gravity vectors (assuming 6 IMUs x 3 components)
        #     current_imu_gravs = current_imu_grav.reshape(6, 3)
        #     prev_imu_gravs = self.prev_imu_grav.reshape(6, 3)
            
        #     # Focus on the X-component of the first IMU (t1_r01) for forward roll direction
        #     current_gravity_x = current_imu_gravs[0, 0]
        #     prev_gravity_x = prev_imu_gravs[0, 0]
        #     orientation_change = current_gravity_x - prev_gravity_x  # Signed change to keep direction
            
        #     # Reward only positive changes (forward roll) above threshold
        #     if orientation_change > 0.57:  # Approximately 30 degrees change in roll (sin(30°) ≈ 0.5)
        #         imu_reward = orientation_change * 10.0  # Scale factor for reward magnitude
            
        #     self.prev_imu_grav = current_imu_grav.copy()
        # else:
        #     self.prev_imu_grav = self._get_IMU_gravity_vectors()

        # Old IMU Reward Term
        # # # Track cumulative roll for continuous rolling reward
        # imu_reward = 0.0
        # if hasattr(self, 'prev_imu_grav'):
        #     current_imu_grav = self._get_IMU_gravity_vectors()
        #     current_imu_gravs = current_imu_grav.reshape(6, 3)
        #     prev_imu_gravs = self.prev_imu_grav.reshape(6, 3)
            
        #     # Calculate roll angle from gravity vector (atan2 gives signed angle)
        #     current_roll = np.arctan2(current_imu_gravs[0, 1], current_imu_gravs[0, 2])  # Y/Z components
        #     prev_roll = np.arctan2(prev_imu_gravs[0, 1], prev_imu_gravs[0, 2])
            
        #     # Handle angle wrapping around ±π
        #     roll_change = current_roll - prev_roll
        #     if roll_change > np.pi:
        #         roll_change -= 2*np.pi
        #     elif roll_change < -np.pi:
        #         roll_change += 2*np.pi
            
        #     # Reward forward rolling (positive direction) with magnitude-based scaling
        #     if roll_change > 0:
        #         imu_reward = roll_change * 50.0  # Scale factor (radians to reward)
        #         # Bonus for sustained rolling speed
        #         if hasattr(self, 'roll_velocity'):
        #             self.roll_velocity = 0.9 * self.roll_velocity + 0.1 * abs(roll_change)
        #             if self.roll_velocity > 0.05:  # ~3 degrees/step sustained
        #                 imu_reward *= 1.5
        #         else:
        #             self.roll_velocity = abs(roll_change)
            
        #     self.prev_imu_grav = current_imu_grav.copy()
        # else:
        #     self.prev_imu_grav = self._get_IMU_gravity_vectors()
        #     self.roll_velocity = 0.0
        
        # Old IMU Reward Term
        # # Reward coordinated rolling across multiple IMUs
        # imu_reward = 0.0
        # if hasattr(self, 'prev_imu_grav'):
        #     current_imu_grav = self._get_IMU_gravity_vectors()
        #     current_imu_gravs = current_imu_grav.reshape(6, 3)
        #     prev_imu_gravs = self.prev_imu_grav.reshape(6, 3)
            
        #     # Calculate orientation changes for multiple IMUs
        #     orientation_changes = []
        #     for i in range(min(3, current_imu_gravs.shape[0])):  # Use first 3 IMUs
        #         # Magnitude of gravity vector change (more robust than single component)
        #         grav_change = np.linalg.norm(current_imu_gravs[i] - prev_imu_gravs[i])
        #         orientation_changes.append(grav_change)
            
        #     avg_change = np.mean(orientation_changes)
        #     change_consistency = 1.0 - np.std(orientation_changes)  # Reward synchronized motion
            
        #     # Reward threshold lowered for realistic step-by-step rolling
        #     if avg_change > 0.05:  # ~3 degree equivalent change
        #         base_reward = avg_change * 20.0
        #         coordination_bonus = max(0, change_consistency) * 10.0
        #         imu_reward = base_reward + coordination_bonus
                
        #         # Additional bonus for forward direction (check center of mass motion)
        #         robot_pos = self.sim.data.qpos[:3]  # Assuming first 3 are robot position
        #         if hasattr(self, 'prev_robot_pos'):
        #             forward_motion = robot_pos[0] - self.prev_robot_pos[0]  # X-direction
        #             if forward_motion > 0:
        #                 imu_reward *= 1.3  # Bonus for forward progress
        #         self.prev_robot_pos = robot_pos.copy()
            
        #     self.prev_imu_grav = current_imu_grav.copy()
        # else:
        #     self.prev_imu_grav = self._get_IMU_gravity_vectors()
        #     self.prev_robot_pos = self.sim.data.qpos[:3].copy()

        # # Reward based on angular velocity patterns consistent with rolling
        # imu_reward = 0.0
        # if hasattr(self, 'prev_imu_grav') and hasattr(self, 'prev_prev_imu_grav'):
        #     current_imu_grav = self._get_IMU_gravity_vectors()
        #     current_imu_gravs = current_imu_grav.reshape(6, 3)
        #     prev_imu_gravs = self.prev_imu_grav.reshape(6, 3)
        #     prev_prev_imu_gravs = self.prev_prev_imu_grav.reshape(6, 3)
            
        #     # Estimate angular velocity from gravity vector changes
        #     # Use first IMU for primary rolling axis detection
        #     grav_vel = (current_imu_gravs[0] - prev_imu_gravs[0])
        #     grav_accel = (current_imu_gravs[0] - 2*prev_imu_gravs[0] + prev_prev_imu_gravs[0])
            
        #     # Rolling motion should show periodic gravity changes
        #     angular_speed = np.linalg.norm(grav_vel)
        #     angular_smoothness = 1.0 / (1.0 + np.linalg.norm(grav_accel))  # Prefer smooth motion
            
        #     # Reward range for realistic rolling speeds
        #     if 0.02 < angular_speed < 0.2:  # Between 1-11 degrees/step
        #         speed_reward = angular_speed * 25.0
        #         smoothness_reward = angular_smoothness * 15.0
        #         imu_reward = speed_reward + smoothness_reward
                
        #         # Check for consistent rolling direction over time
        #         if hasattr(self, 'rolling_direction_history'):
        #             current_direction = np.sign(np.dot(grav_vel, [1, 0, 0]))  # X-axis preference
        #             self.rolling_direction_history.append(current_direction)
        #             if len(self.rolling_direction_history) > 10:
        #                 self.rolling_direction_history.pop(0)
                    
        #             # Bonus for consistent rolling direction
        #             direction_consistency = abs(np.mean(self.rolling_direction_history))
        #             if direction_consistency > 0.6:  # 60% consistency
        #                 imu_reward *= (1.0 + direction_consistency)
        #         else:
        #             self.rolling_direction_history = []
            
        #     # Update history
        #     self.prev_prev_imu_grav = self.prev_imu_grav.copy()
        #     self.prev_imu_grav = current_imu_grav.copy()
        # else:
        #     current_imu_grav = self._get_IMU_gravity_vectors()
        #     if hasattr(self, 'prev_imu_grav'):
        #         self.prev_prev_imu_grav = self.prev_imu_grav.copy()
        #     else:
        #         self.prev_prev_imu_grav = current_imu_grav.copy()
        #     self.prev_imu_grav = current_imu_grav.copy()
        #     self.rolling_direction_history = []


        imu_x_rotation_reward = self._reward_x_axis_rotation()

        imu_x_rotation_speed_reward = self._reward_x_axis_desired_rotation_speed(desired_speed=0.5)

        # Weighting individual reward components
        # velocity_reward *= 0.0
        # distance_reward *= 0.0 
        # imu_x_rotation_reward *= 10.0 #as 1 one roll
        # imu_x_rotation_speed_reward *= 10.0 #as 10 one roll
        # penalties *= 0.0  #Making this 1 did eliminate the oscillations, which is good

        # reward = velocity_reward + distance_reward + penalties + imu_x_rotation_reward + imu_x_rotation_speed_reward
        
        # In your sim_step() method, replace the oscillating rewards:

        # OLD - Promotes oscillations
        # imu_x_rotation_reward = self._reward_x_axis_rotation()
        # imu_x_rotation_speed_reward = self._reward_x_axis_desired_rotation_speed(desired_speed=0.5)


        # NOTE: Hey Zac, i thought we weren't incentivizing actual rolling, so the jittering stopped but weird behaviour here
        # NEW - Promotes actual rolling
        cumulative_rotation_reward = self._reward_cumulative_x_axis_rotation()
        consistent_direction_reward = self._reward_consistent_rolling_direction(window_size=15)
        displacement_progress_reward = self._reward_angular_displacement_progress(target_rotations_per_episode=1)

        # Weighting - adjust based on what works best
        cumulative_rotation_reward *= 5.0      # Reward total rotation progress
        consistent_direction_reward *= 3.0     # Reward consistent direction
        displacement_progress_reward *= 10.0    # Reward actual angular displacement

        reward = (velocity_reward + distance_reward + penalties*2.0 + 
          cumulative_rotation_reward + consistent_direction_reward + displacement_progress_reward)
        # # Calculate total forward distance reward
        # if hasattr(self, 'prev_pos') and self.prev_pos is not None:
        #     # Calculate displacement from initial state

        self.prev_pos = robot_pos.copy()
        self.step_count = getattr(self, 'step_count', 0) + 1

        # Build observation based on selected mode
        observation = self.get_observation()
        
        done = False
        info = {}
        
        return observation, reward, done, info

    def get_robot_position(self):
        """
        Returns the current position of the robot.
        """
        self.forward()
        debug_print(f"Tensegrity positions: {self.mjc_data.qpos}", "tensegrity_mjc_simulation.py", self.debug_enabled)
        return self.mjc_data.qpos[:3]  # Assuming the first three elements represent the robot's position

    def _reward_x_axis_rotation(self):
        """
        Reward term that encourages rotation about the x-axis.
        Returns higher reward for faster x-axis rotation across all IMUs.
        """
        imu_ang = self._get_IMU_angular_velocities()
        
        # Extract x-axis angular velocities (every 3rd element starting from index 0)
        x_angular_vels = imu_ang[::3]  # [0, 3, 6, ...] - x components
        
        # Reward based on absolute x-axis angular velocity
        # Use absolute value to reward rotation in either direction
        x_rotation_reward = np.sum(np.abs(x_angular_vels))
        
        # Optional: Add scaling factor to tune reward magnitude
        return x_rotation_reward
    
    def _reward_x_axis_desired_rotation_speed(self, desired_speed=1.0):
        """
        Reward term that encourages rotation about the x-axis at a desired speed.
        Returns higher reward for x-axis rotation velocities closer to the desired speed.
        """
        imu_ang = self._get_IMU_angular_velocities()
        
        # Extract x-axis angular velocities (every 3rd element starting from index 0)
        x_angular_vels = imu_ang[::3]
        # Reward based on closeness to desired speed
        speed_diff = np.abs(x_angular_vels - desired_speed)
        x_rotation_reward = np.sum(np.maximum(0, 1.0 - speed_diff))  # Reward decreases with speed difference
        return x_rotation_reward
    
    def _reward_cumulative_x_axis_rotation(self):
        """
        Reward term that encourages sustained rolling by tracking cumulative rotation.
        Prevents oscillation rewards by only rewarding net rotational progress.
        """
        # Get current IMU orientations (quaternions or Euler angles)
        current_orientations = self._get_IMU_orientations()  # You may need to implement this
        
        # Initialize rotation tracking if first call
        if not hasattr(self, 'prev_x_rotations'):
            self.prev_x_rotations = np.zeros(len(current_orientations) // 3)  # Assuming 3 values per IMU
            self.cumulative_x_rotations = np.zeros(len(current_orientations) // 3)
            return 0.0
        
        # Extract x-axis rotations (first component of each IMU's orientation)
        current_x_rotations = current_orientations[::3]  # was {::3} [0, 3, 6, ...] - x components
        
        # Calculate rotation differences (handle angle wrapping)
        rotation_diffs = []
        for i, (curr, prev) in enumerate(zip(current_x_rotations, self.prev_x_rotations)):
            # Handle angle wrapping (-π to π)
            diff = curr - prev
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            rotation_diffs.append(diff)
        
        rotation_diffs = np.array(rotation_diffs)
        
        # Update cumulative rotations
        self.cumulative_x_rotations += rotation_diffs
        
        # Reward based on cumulative rotation magnitude
        cumulative_reward = np.sum(np.abs(self.cumulative_x_rotations))
        
        # Update previous rotations for next call
        self.prev_x_rotations = current_x_rotations.copy()
        
        return cumulative_reward
    
    def _reward_consistent_rolling_direction(self, window_size=10):
        """
        Reward term that encourages consistent rolling direction over time.
        Penalizes direction changes that indicate oscillation rather than rolling.
        """
        imu_ang = self._get_IMU_angular_velocities()
        x_angular_vels = imu_ang[::3]  # [0, 3, 6, ...] - x components
        
        # Initialize rolling direction history if needed
        if not hasattr(self, 'rolling_direction_history'):
            self.rolling_direction_history = []
        
        # Determine current rolling direction (positive/negative/stopped)
        avg_x_velocity = np.mean(x_angular_vels)
        if abs(avg_x_velocity) > 0.1:  # Threshold to avoid noise
            current_direction = 1 if avg_x_velocity > 0 else -1
        else:
            current_direction = 0  # Not rolling
        
        # Add to history and maintain window size
        self.rolling_direction_history.append(current_direction)
        if len(self.rolling_direction_history) > window_size:
            self.rolling_direction_history.pop(0)
        
        # Calculate consistency reward
        if len(self.rolling_direction_history) >= window_size:
            # Count direction changes (oscillations)
            direction_changes = 0
            for i in range(1, len(self.rolling_direction_history)):
                if (self.rolling_direction_history[i] != 0 and 
                    self.rolling_direction_history[i-1] != 0 and
                    self.rolling_direction_history[i] != self.rolling_direction_history[i-1]):
                    direction_changes += 1
            
            # Reward consistency (fewer direction changes)
            consistency_reward = max(0, window_size - direction_changes * 2)
            
            # Bonus for sustained rolling in one direction
            non_zero_directions = [d for d in self.rolling_direction_history if d != 0]
            if len(non_zero_directions) >= window_size * 0.8:  # 80% of time spent rolling
                if len(set(non_zero_directions)) == 1:  # All same direction
                    consistency_reward += 10.0  # Sustained rolling bonus
            
            return consistency_reward
        
        return 0.0
    
    def _reward_angular_displacement_progress(self, target_rotations_per_episode=2):
        """
        Reward based on total angular displacement from start position.
        Encourages actual rolling progress rather than oscillations.
        """
        # Get current robot orientation
        robot_quat = self.mjc_data.qpos[3:7]  # Quaternion orientation
        
        # Initialize starting orientation if first call
        if not hasattr(self, 'initial_orientation'):
            self.initial_orientation = robot_quat.copy()
            return 0.0
        
        # Calculate total rotation from initial orientation
        # Convert quaternions to rotation matrices and compute angle difference
        from scipy.spatial.transform import Rotation as R
        
        initial_rot = R.from_quat(self.initial_orientation)
        current_rot = R.from_quat(robot_quat)
        
        # Calculate relative rotation
        relative_rot = current_rot * initial_rot.inv()
        
        # Extract x-axis rotation component
        euler_angles = relative_rot.as_euler('xyz')
        x_rotation_progress = abs(euler_angles[0])  # 0 is X-axis rotation in radians
        
        # Reward based on progress toward target rotations
        target_radians = target_rotations_per_episode * 2 * np.pi
        progress_reward = min(x_rotation_progress / target_radians, 1.0) * 100  # Scale to 0-100
        
        return progress_reward
    
    def _get_IMU_orientations(self):
        """
        Get IMU orientation data from MuJoCo sensors.
        Returns orientations for all IMU sensors in the simulation.
        """
        orientations = []
        
        # Following coding guidelines - verify against actual sensor names in XML
        try:
            # Attempt to get orientation data from IMU sensors
            for i in range(6):  # Assuming 6 IMUs based on coding guidelines
                try:
                    # Try different sensor naming conventions
                    sensor_names = [f"imu_{i}", f"angvel_{i}", f"orientation_{i}"]
                    
                    for sensor_name in sensor_names:
                        try:
                            orientation_data = self.mjc_data.sensor(sensor_name).data
                            orientations.extend(orientation_data[:3])  # First 3 components
                            break
                        except:
                            continue
                    else:
                        # If no sensor found, use body orientation
                        if i < len(self.mjc_model.body_names):
                            body_quat = self.mjc_data.body(i).xquat
                            # Convert quaternion to Euler for x-component
                            from scipy.spatial.transform import Rotation as R
                            euler = R.from_quat(body_quat).as_euler('xyz')
                            orientations.extend(euler)
                except:
                    # Fallback: use zeros if sensor access fails
                    orientations.extend([0.0, 0.0, 0.0])
        
        except Exception as e:
            debug_print(f"Warning: Could not get IMU orientations: {e}", 
                    "tensegrity_mjc_simulation.py", self.debug_enabled)
            # Fallback to angular velocities as proxy
            return self._get_IMU_angular_velocities()
        
        return np.array(orientations)

    def calculate_omnidirectional_distance_reward(self, robot_pos):
        """Reward for exploring the XY plane in any direction"""
        
        # Initialize tracking variables
        if not hasattr(self, 'max_distance_from_origin'):
            self.max_distance_from_origin = 0.0
            self.origin_pos = robot_pos[:2].copy()  # Store starting position
        
        # Calculate distance from origin in XY plane
        xy_distance_from_origin = np.linalg.norm(robot_pos[:2] - self.origin_pos)
        
        # Reward for reaching new maximum distances
        exploration_reward = 0.0
        if xy_distance_from_origin > self.max_distance_from_origin:
            exploration_reward = (xy_distance_from_origin - self.max_distance_from_origin) * 1000.0
            self.max_distance_from_origin = xy_distance_from_origin
        
        # Base distance reward (encourages staying away from origin)
        base_distance_reward = xy_distance_from_origin

        return exploration_reward + 0.0 * base_distance_reward 

    def get_endpts(self):
        # Get end point xyz coordinates
        end_pts = []
        for end_pt_site in self.end_pts:
            end_pt = self.mjc_data.sensor(f"pos_{end_pt_site}").data
            end_pts.append(end_pt)

        end_pts = np.vstack(end_pts)
        return end_pts
    
    def render(self, mode='human', width=None, height=None):
        """Render the simulation"""
        try:
            # Import here to avoid issues
            import mujoco
            
            # Use parent class renderer if available, otherwise create one
            if hasattr(self, 'renderer') and self.renderer is not None:
                # Use the properly configured renderer from parent class
                self.renderer.update_scene(self.mjc_data)
                frame = self.renderer.render()
            else:
                # Fallback: create renderer if it doesn't exist
                if not hasattr(self, '_renderer'):
                    # Use provided dimensions or defaults
                    if width is None or height is None:
                        width = 800
                        height = 600
                    self._renderer = mujoco.Renderer(self.mjc_model, height, width)
                
                # Update scene and render
                self._renderer.update_scene(self.mjc_data)
                frame = self._renderer.render()
            
            if mode == 'human':
                import cv2
                # Convert RGB to BGR for OpenCV display
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('Tensegrity Robot', frame_bgr)
                cv2.waitKey(1)
            
            return frame
            
        except Exception as e:
            debug_print(f"Rendering failed: {e}", "tensegrity_mjc_simulation.py", self.debug_enabled)
            return None

    def close(self):
        """Close renderer and cleanup"""
        try:
            if hasattr(self, '_renderer'):
                self._renderer.close()
            import cv2
            cv2.destroyAllWindows()
        except:
            pass

    def apply_random_perturbation(self):
        """Apply random forces/torques to prevent exploitation"""
        if self.step_count % 50 == 0:  # Every 50 steps
            # Random external force on platform
            force_magnitude = np.random.uniform(0.5, 2.0)
            force_direction = np.random.uniform(-1, 1, 3)  # Random x,y,z direction
            force_direction = force_direction / np.linalg.norm(force_direction)
            
            # Apply force to platform body
            platform_body_id = mujoco.mj_name2id(self.mjc_model, mujoco.mjtObj.mjOBJ_BODY, "t1_r01")
            if platform_body_id >= 0:
                self.mjc_data.xfrc_applied[platform_body_id][:3] = force_magnitude * force_direction

    def calculate_anti_exploit_penalties(self, robot_pos, controls):
        """Penalties to prevent common exploitation behaviors"""
        penalties = 0.0
        
        # 1. Prevent excessive bouncing (z-axis exploitation)
        if hasattr(self, 'prev_pos') and self.prev_pos is not None:
            z_velocity = abs((robot_pos[2] - self.prev_pos[2]) / self.dt)
            if z_velocity > 1.0:  # Too much vertical movement
                penalties -= z_velocity * 5.0
        
        # 2. Prevent rapid oscillations in controls
        if hasattr(self, 'prev_controls'):
            control_change = np.sum(np.abs(controls - self.prev_controls))
            if control_change > 2.0:  # Too rapid changes
                penalties -= control_change * 2.0
        self.prev_controls = controls.copy()
        
        # 3. Energy efficiency penalty
        # energy_cost = np.sum(np.abs(controls)) * 0.01
        # penalties -= energy_cost
        
        # 4. FIX: Use correct body name for stability check
        # Choose one of the robot bodies (e.g., t1_r01) for orientation check
        # try:
        #     platform_quat = self.mjc_data.body("t1_r01").xquat  # Use actual body name
        #     tilt_penalty = abs(platform_quat[1]) + abs(platform_quat[2])  # Penalize roll/pitch
        #     penalties -= tilt_penalty * 10.0
        # except KeyError:
        #     # Fallback: skip tilt penalty if body not found
        #     pass

        penalties = penalties # * 0.1

        return penalties

    def get_observation(self):
        """Dispatch observation based on obs_mode."""
        if self.obs_mode == "tier2":
            return self.get_observation_tier2()
        return self.get_observation_legacy104()

    def get_observation_legacy104(self):
        """
        Legacy comprehensive observation
        Returns 104-dimensional observation vector (42 robot state + 24 cable + 36 endpoints + padding)
        """
        observation = []
        
        # 1. Robot state (42 dims): positions + orientations of all bodies
        robot_state = self.get_robot_state()
        observation.extend(robot_state)
        
        # 2. Cable state (24 dims): current lengths + velocities
        cable_state = self.get_cable_state()
        observation.extend(cable_state)
        
        # 3. End effector positions (18 dims): spatial locations
        end_effector_state = self.get_end_effector_state()
        observation.extend(end_effector_state)
        
        # # 4. Time-based features (2 dims): simulation time + step count
        # sim_time = self.mjc_data.time
        # step_normalized = (getattr(self, 'step_count', 0) % 1000) / 1000.0  # Normalize step count
        # observation.extend([sim_time, step_normalized])
        
        # Verify dimension consistency
        obs_array = np.array(observation, dtype=np.float32)
        expected_dim = 104
        
        if len(obs_array) != expected_dim:
            debug_print(f"⚠️ OBSERVATION DIMENSION MISMATCH: Got {len(obs_array)}, expected {expected_dim}", "tensegrity_mjc_simulation.py", self.debug_enabled)
            # Pad or truncate to match expected dimension
            if len(obs_array) < expected_dim:
                obs_array = np.pad(obs_array, (0, expected_dim - len(obs_array)))
            else:
                obs_array = obs_array[:expected_dim]
        
        return obs_array

    def get_robot_state(self):
        """Get robot body positions and orientations"""
        robot_state = []
        robot_bodies = ["t1_r01", "t1_r23", "t1_r45", "t2_r01", "t2_r23", "t2_r45"]
        
        for body_name in robot_bodies:
            try:
                pos = self.mjc_data.body(body_name).xpos  # 3D position
                quat = self.mjc_data.body(body_name).xquat  # 4D quaternion
                robot_state.extend(pos)
                robot_state.extend(quat)
            except KeyError:
                robot_state.extend([0.0] * 7)  # Fallback if body not found
        
        return robot_state

    def get_cable_state(self):
        """Get current cable lengths and velocities"""
        cable_state = []
        
        # Initialize previous cable lengths tracking
        if not hasattr(self, 'prev_cable_lengths'):
            self.prev_cable_lengths = [0.0] * self.n_actuators
        
        current_lengths = []
        for i in range(self.n_actuators):
            # Get current cable length using your existing method
            if hasattr(self, 'cable_sites') and i < len(self.cable_sites):
                try:
                    s0 = self.mjc_data.sensor(f"pos_{self.cable_sites[i][0]}").data
                    s1 = self.mjc_data.sensor(f"pos_{self.cable_sites[i][1]}").data
                    current_length = np.linalg.norm(s1 - s0)
                except:
                    current_length = 0.0
            else:
                current_length = 0.0
            
            current_lengths.append(current_length)
            
            # Calculate length velocity
            length_velocity = (current_length - self.prev_cable_lengths[i]) / self.dt
            
            cable_state.extend([current_length, length_velocity])
        
        # Update previous lengths for next step
        self.prev_cable_lengths = current_lengths
        
        return cable_state

    def get_end_effector_state(self):
        """Get positions of robot end points"""
        try:
            end_pts = self.get_endpts()  # Use existing method
            return end_pts.flatten()
        except:
            return [0.0] * 18  # 6 endpoints × 3D fallback

    # --------------------------- Tier-2 Observation ---------------------------
    def _get_actuated_cable_lengths(self):
        lengths = []
        for idx in self.actuated_ids:
            try:
                sites = self.cable_sites[idx]
                s0 = self.mjc_data.sensor(f"pos_{sites[0]}").data
                s1 = self.mjc_data.sensor(f"pos_{sites[1]}").data
                lengths.append(np.linalg.norm(s1 - s0))
            except Exception:
                lengths.append(0.0)
        return np.asarray(lengths, dtype=np.float32)

    def _get_strain_extensions(self):
        # (current_length - rest_length) / rest_length for selected tendons
        exts = []
        for tid in self.strain_tendon_ids:
            try:
                sites = self.cable_sites[tid]
                s0 = self.mjc_data.sensor(f"pos_{sites[0]}").data
                s1 = self.mjc_data.sensor(f"pos_{sites[1]}").data
                curr = np.linalg.norm(s1 - s0)
                rest = self.mjc_model.tendon_lengthspring[tid, 0]
                val = 0.0 if rest <= 1e-6 else (curr - rest) / rest
                exts.append(val)
            except Exception:
                exts.append(0.0)
        return np.clip(np.asarray(exts, dtype=np.float32), -1.0, 2.0)

    def _compute_COM_position(self):
        eps = self.get_endpts()
        return eps.mean(axis=0)

    def _compute_COM_velocities(self):
        curr_COM = self._compute_COM_position()
        if self.prev_COM_pos is None:
            lin_vel = np.zeros(3, dtype=np.float32)
        else:
            lin_vel = (curr_COM - self.prev_COM_pos) / self.dt
        self.prev_COM_pos = curr_COM

        # Angular velocity: use global qvel rotational components if available
        if self.mjc_data.qvel.shape[0] >= 3:
            ang_vel = self.mjc_data.qvel[:3].copy()
        else:
            ang_vel = np.zeros(3, dtype=np.float32)
        return lin_vel.astype(np.float32), ang_vel.astype(np.float32)

    def _quat_to_rot(self, quat):
        # MuJoCo quaternions: [w,x,y,z]
        w, x, y, z = quat
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ], dtype=np.float32)

    def _get_IMU_gravity_vectors(self):
        grav = np.array([0, 0, -1.0], dtype=np.float32)
        vecs = []
        for g in self.imu_geom_names:
            try:
                quat = self.mjc_data.sensor(f"quat_{g}").data  # [w,x,y,z]
                R = self._quat_to_rot(quat)
                g_local = R.T @ grav
                vecs.append(g_local.astype(np.float32))
            except Exception:
                vecs.append(np.zeros(3, dtype=np.float32))
        return np.asarray(vecs, dtype=np.float32).reshape(-1)

    def _get_IMU_angular_velocities(self):
        angs = []
        for g in self.imu_geom_names:
            try:
                ang = self.mjc_data.sensor(f"angvel_{g}").data
                angs.append(ang.astype(np.float32))
            except Exception:
                angs.append(np.zeros(3, dtype=np.float32))
        return np.asarray(angs, dtype=np.float32).reshape(-1)

    def get_observation_tier2(self):
        """Tier-2 96D observation without contacts/accelerometers."""
        # 1. Cable lengths (normalized)
        cable_lengths = self._get_actuated_cable_lengths()
        denom = max(self.max_cable_length - self.min_cable_length, 1e-6)
        cable_lengths_norm = np.clip((cable_lengths - self.min_cable_length) / denom, 0.0, 1.0)

        # 2. Cable length rates (normalized finite diff)
        cable_rates = (cable_lengths - self.prev_cable_lengths) / self.dt
        rate_scale = denom
        cable_rates_norm = np.clip(cable_rates / rate_scale, -1.0, 1.0)
        self.prev_cable_lengths = cable_lengths.copy()

        # 3. Previous action (normalized targets)
        prev_action = self.prev_action.copy()

        # 4. Strain sensor normalized extensions (18)
        strain_exts = self._get_strain_extensions()

        # 5. IMU gravity vectors (6*3)
        imu_grav = self._get_IMU_gravity_vectors()

        # 6. IMU angular velocities (6*3), scaled
        imu_ang = self._get_IMU_angular_velocities()
        imu_ang_norm = np.clip(imu_ang / 10.0, -1.0, 1.0)

        # 7 & 8. COM linear and angular velocities (scaled)
        com_lin_vel, com_ang_vel = self._compute_COM_velocities()
        com_lin_vel_norm = np.clip(com_lin_vel / 1.0, -3.0, 3.0)
        com_ang_vel_norm = np.clip(com_ang_vel / 10.0, -1.0, 1.0)

        parts = [
            cable_lengths_norm,          # 12
            cable_rates_norm,            # 12
            prev_action,                 # 12
            strain_exts,                 # 18
            imu_grav,                    # 18
            imu_ang_norm,                # 18
            com_lin_vel_norm,            # 3
            com_ang_vel_norm             # 3
        ]
        obs = np.concatenate(parts).astype(np.float32)
        # Ensure exact 96 dims
        if obs.shape[0] != 96:
            if obs.shape[0] < 96:
                obs = np.pad(obs, (0, 96 - obs.shape[0]))
            else:
                obs = obs[:96]
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)


class MultiProcTensegrityMujocoSimulator:

    def __init__(self,
                 num_sims,
                 xml_path: Path,
                 # visualize: bool = True,
                 render_size: tuple = (240, 240),
                 render_fps: int = 100,
                 num_actuated_cables: int = 12,
                 num_rods: int = 3):
        self.sims = [
            TensegrityMuJoCoSimulator(
                xml_path,
                False,
                # visualize,
                render_size,
                render_fps,
                num_actuated_cables,
                num_rods
            )
            for _ in range(num_sims)
        ]

    def reset(self):
        # for sim in self.sims:
        #     sim.reset()
        """
        Resets the robots as if it was just instantiated from the xml file.
        """
        super().reset()
        self.bring_to_grnd()

        for motor in self.cable_motors:
            motor.reset_omega_t()
        
        # Initialize RL state
        self.prev_pos = None
        self.step_count = 0
        
        # Return initial observation
        return self._get_observation()

    def set_state(self, all_states: np.ndarray):
        assert len(self.sims) == all_states.shape[0], "Number of sims does not match number of states"

        for i, sim in enumerate(self.sims):
            state = all_states[i].reshape(-1, 13)
            sim.mjc_data.qpos = state[:, :7].flatten()
            sim.mjc_data.qvel = state[:, 7:].flatten()

    def get_poses(self):
        return np.stack([sim.mjc_data.qpos for sim in self.sims], dim=0)

    def get_vels(self):
        return np.stack([sim.mjc_data.qvel for sim in self.sims], dim=0)

    def get_frames(self):
        return [sim.render_frame() for sim in self.sims]

    def _sim_step(self, sim, controls, queue, idx):
        sim.sim_step(controls)
        queue.put(True)

    def _run_target_lengths(self, sim, target_lengths, queue, idx):
        sim.run_target_lengths(target_lengths)
        queue.put(True)

    def _parallel_proc(self, input_args, proc_fn, max_num_parallel):
        num_procs_ran = 0
        while num_procs_ran < len(input_args):
            num_parallel = min(max_num_parallel, len(input_args) - num_procs_ran)
            queues = [multiprocessing.Queue() for _ in range(num_parallel)]
            processes = [
                multiprocessing.Process(
                    target=proc_fn,
                    args=(self.sims[i], input_args[i], queues[i], i)
                ) for i in range(num_parallel)
            ]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            status = [False for _ in range(num_parallel)]
            while not all(status):
                for i in range(num_parallel):
                    status[i] = queues[i].get()

            num_procs_ran += num_parallel

    def parallel_sim_step(self, controls: np.ndarray, max_num_parallel: int = 10):
        self._parallel_proc(controls, self._sim_step, max_num_parallel)

    def parallel_run_target_lengths(self, target_lengths: List, max_num_parallel: int = 10):
        self._parallel_proc(target_lengths, self._run_target_lengths, max_num_parallel)

    # Add this method to the TensegrityMuJoCoSimulator class
    # Add this method to your TensegrityMuJoSimulator class (at the end of the class)
