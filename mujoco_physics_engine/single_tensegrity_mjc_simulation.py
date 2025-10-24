"""Single tensegrity MuJoCo simulator.

This module provides a single 3-bar tensegrity robot simulator for the
`xml_models/3bar_new_platform_all_cables.xml` model.

Key specifications:
- 6 actuated cables (tendons td_0 to td_5 with default stiffness 100000)
- 9 passive cables (tendons td_6 to td_14 with explicit stiffness values)
- 3 rods for IMU sensing (r01, r23, r45)
- Observation dimension: flexible (default 27 = 6 lengths + 6 rates + 6 prev_action + 9 IMU)
- Actions: normalized target cable lengths in [0,1]
- Reward: copied from dual robot with dimension adjustments
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import mujoco
import time

from .mujoco_simulation import AbstractMuJoCoSimulator
from .pid import PID
from .cable_motor import DCMotor


def debug_print(msg: str, filename: str = "single_tensegrity_mjc_simulation.py", enabled: bool = False):
    """Print debug messages with filename prefix if debug is enabled"""
    if enabled:
        print(f"DEBUG {filename}: {msg}")


class SingleTensegrityMuJoCoSimulator(AbstractMuJoCoSimulator):
    """Single tensegrity simulator with 6 actuated cables.
    
    Action: np.ndarray shape (6,) normalized target lengths in [0,1].
    Observation: np.ndarray shape (obs_dim,) where obs_dim is calculated from components.
    """

    def __init__(
        self,
        xml_path: Path | str = Path("mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml"),
        visualize: bool = True,
        render_size: Tuple[int, int] = (720, 720),
        render_fps: int = 20,
        obs_dim: Optional[int] = None,  # If None, calculate from components
        debug_enabled: bool = False,
        controller_kp: float = 10.0,
        controller_ki: float = 0.2,
        controller_kd: float = 2.0,
        min_cable_length: float = 0.6,
        max_cable_length: float = 1.6,
        control_penalty_cap: float = 100.0,
        render_pause: float = 0.01,  # Pause between renders in seconds (for visualization speed control)
    ):
        super().__init__(Path(xml_path), visualize, render_size, render_fps)
        
        self.debug_enabled = debug_enabled
        self.control_penalty_cap = float(control_penalty_cap)
        self.min_cable_length = min_cable_length
        self.max_cable_length = max_cable_length
        self.render_pause = render_pause  # Store pause duration
        
        # NEEDS UPDATE: Verify these are the correct actuated tendon IDs for the single robot XML
        # Based on XML analysis: td_0 to td_5 are the first 6 tendons with default stiffness 100000
        self.actuated_ids = [0, 1, 2, 3, 4, 5]  # First 6 tendons are actuated
        self.n_actuators = len(self.actuated_ids)
        
        # HARDCODED CABLE SITE PAIRS from XML analysis
        # Mapping td_0 to td_5 to their site pairs
        self.cable_sites = [
            # td_0
            ("s_3_b5", "s_b5_3"),
            # td_1
            ("s_1_b3", "s_b3_1"),
            # td_2
            ("s_5_b1", "s_b1_5"),
            # td_3
            ("s_0_b2", "s_b2_0"),
            # td_4
            ("s_4_b0", "s_b0_4"),
            # td_5
            ("s_2_b4", "s_b4_2"),
        ]
        
        # PID controllers and DC motors for each actuator
        self.pids = [
            PID(Kp=controller_kp, Ki=controller_ki, Kd=controller_kd, dt=self.dt, debug_enabled=self.debug_enabled)
            for _ in range(self.n_actuators)
        ]
        self.cable_motors = [DCMotor(debug_enabled=self.debug_enabled) for _ in range(self.n_actuators)]
        
        # Observation components (flexible configuration)
        # Default: 6 cable_lengths + 6 cable_rates + 6 prev_action + 9 IMU (3 rods × 3D) = 27
        self.obs_components = {
            'cable_lengths': 6,  # normalized [0,1]
            'cable_rates': 6,    # normalized rates
            'prev_action': 6,    # previous action
            'imu': 9,            # 3 rods × 3D gravity vectors
        }
        
        # Calculate observation dimension
        if obs_dim is None:
            self.obs_dim = sum(self.obs_components.values())
        else:
            self.obs_dim = obs_dim
        
        # Rod names for IMU (3 rods in single robot)
        self.imu_geom_names = ["r01", "r23", "r45"]
        self.n_rods = len(self.imu_geom_names)
        
        # Endpoint sites (vertices s0-s5)
        self.end_pts = ["s0", "s1", "s2", "s3", "s4", "s5"]
        
        # State tracking buffers
        self.prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        self.prev_lengths = np.zeros(self.n_actuators, dtype=np.float32)
        self.prev_pos = None
        self.prev_controls = None
        self.step_count = 0
        
        # Viewer for real-time visualization (lazy-loaded in render())
        self._viewer = None
        
        # Initialize visualization camera if needed
        if self.visualize and hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.cam.distance = 12.0  # Zoom out for single robot
            self.viewer.cam.lookat[0] = 0.0
        
        debug_print(f"Initialized SingleTensegrityMuJoCoSimulator", "single_tensegrity_mjc_simulation.py", self.debug_enabled)
        debug_print(f"  Actuators: {self.n_actuators}", "single_tensegrity_mjc_simulation.py", self.debug_enabled)
        debug_print(f"  Observation dim: {self.obs_dim}", "single_tensegrity_mjc_simulation.py", self.debug_enabled)
        debug_print(f"  Cable sites: {len(self.cable_sites)}", "single_tensegrity_mjc_simulation.py", self.debug_enabled)

    def reset(self):
        """Reset simulator to initial state."""
        super().reset()
        self.forward()
        
        # Reset tracking variables
        self.prev_lengths = self._get_actuated_cable_lengths()
        self.prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        self.prev_pos = None
        self.prev_controls = None
        self.step_count = 0
        
        # Reset reward tracking variables (matching dual robot)
        if hasattr(self, 'ground_state'):
            delattr(self, 'ground_state')
        if hasattr(self, 'lift_state'):
            delattr(self, 'lift_state')
        if hasattr(self, 'prev_ground_mask'):
            delattr(self, 'prev_ground_mask')
        if hasattr(self, 'ground_duty_ema'):
            delattr(self, 'ground_duty_ema')
        if hasattr(self, 'lifted_duty_ema'):
            delattr(self, 'lifted_duty_ema')
        if hasattr(self, 'prev_lifted_mask'):
            delattr(self, 'prev_lifted_mask')
        if hasattr(self, 'prev_lifted_centroid_xy'):
            delattr(self, 'prev_lifted_centroid_xy')
        if hasattr(self, 'prev_grounded_centroid_xy'):
            delattr(self, 'prev_grounded_centroid_xy')
        if hasattr(self, 'low_speed_hist'):
            delattr(self, 'low_speed_hist')
        if hasattr(self, 'stall_streak'):
            delattr(self, 'stall_streak')
        if hasattr(self, 'high_speed_hist'):
            delattr(self, 'high_speed_hist')
        if hasattr(self, 'lift_streaks'):
            delattr(self, 'lift_streaks')
        if hasattr(self, 'prev_end_pts'):
            delattr(self, 'prev_end_pts')
        
        return self.get_observation()

    def sim_step(self, target_lengths=None):
        """Single environment step applying target normalized cable lengths.
        
        Reward function copied from dual robot with dimension adjustments.
        
        Parameters
        ----------
        target_lengths : np.ndarray, shape (6,)
            Normalized target cable lengths in [0, 1]
        
        Returns
        -------
        observation : np.ndarray
        reward : float
        done : bool
        info : dict
        """
        # Compute physics timesteps per action
        steps_per_action = int(1.0 / self.dt)  # 100 steps for dt=0.01

        # Control signals for each actuator
        controls = None

        # Check that target lengths are provided
        if target_lengths is not None:

            # Normalize and cache last action for observations
            try:
                desired_norms = np.asarray(target_lengths, dtype=float)
            except Exception:
                desired_norms = np.array(list(target_lengths), dtype=float)
            # Enforce [0,1] without raising to keep sim running
            desired_norms = np.clip(desired_norms, 0.0, 1.0)
            # Update previous action buffer used by observations
            if hasattr(self, 'prev_action'):
                self.prev_action = desired_norms.astype(np.float32)

            # Iterate over timesteps to apply motor controls and step physics
            for step_i in range(steps_per_action):

                # Initialize control signals for this step
                controls = np.zeros(self.n_actuators, dtype=float)

                # Compute control signals for each actuator for this step
                for i in range(len(target_lengths)):
                    # lengths = target_lengths[i]
                    # Get current rest length
                    rest_length = self.mjc_model.tendon_lengthspring[self.actuated_ids[i], 0]
                    # Get desired normalized length
                    desired_norm = float(desired_norms[i])
                    # Calculate current length
                    s0 = self.mjc_data.sensor(f"pos_{self.cable_sites[self.actuated_ids[i]][0]}").data
                    s1 = self.mjc_data.sensor(f"pos_{self.cable_sites[self.actuated_ids[i]][1]}").data
                    curr_length = np.linalg.norm(s1 - s0)
                    # Calculate control action
                    ctrl, _ = self.pids[i].update_control_by_target_norm_length(
                        curr_length, desired_norm, rest_length,
                        self.min_cable_length, self.max_cable_length
                    )
                    # Store control action
                    controls[i] = -1.0 * ctrl
                    # Mirror subset (domain-specific constraint)
                    if self.n_actuators >= 9:
                        for i in range(3):
                            if i+6 < self.n_actuators and i+3 < self.n_actuators:
                                controls[i+6] = controls[i+3]

                # Update tendon rest lengths based on control action
                for tendon_id in self.actuated_ids:
                    # Find corresponding action index
                    action_idx = self.actuated_ids.index(tendon_id)
                    # Get the correct control for the current tendon
                    ctrl = controls[action_idx]
                    # Get the current rest length
                    cable_rest_length = self.mjc_model.tendon_lengthspring[tendon_id, 0]
                    # Compute the change in rest length due to the control action
                    dl = self.cable_motors[action_idx].compute_cable_length_delta(ctrl, self.dt)
                    # Update the rest length, ensuring it stays within bounds
                    new_rest_length = np.clip(cable_rest_length + dl, self.min_cable_length, self.max_cable_length)
                    # Apply the new rest length to the MuJoCo model
                    self.mjc_model.tendon_lengthspring[tendon_id] = new_rest_length
                    # Debug print
                    if step_i == 0:  # Only debug print first iteration to avoid spam
                        debug_print(f"Cable {tendon_id} dl={dl:.4f} ctrl={ctrl:.3f} newL={new_rest_length:.3f}",
                                    "tensegrity_mjc_simulation.py", self.debug_enabled)

                # Step physics once all controls are set
                mujoco.mj_step(self.mjc_model, self.mjc_data)
                self.forward()
                
                # Render if visualization is enabled (smooth rendering during action execution)
                if self.visualize and step_i % 1 == 0:  # Render every 5th physics step to balance smoothness and performance
                    self.render()
                    time.sleep(self.render_pause)  # Pause to slow down visualization

        else:
            # Raise critical error if no target lengths provided
            print("Target lengths must be provided for sim_step. Controls set to zero.")
            controls = np.zeros(self.n_actuators, dtype=float)
        
        # ===== REWARD COMPUTATION (copied from dual robot) =====
        
        # Get end points for locomotion reward
        end_pts = self.get_endpts()
        robot_pos = end_pts.mean(axis=0)  # Use mean of endpoints as robot position
        
        # Endpoint Height Reward: anti-glide via turnover, duty cycle, and lift transitions
        end_pts_z = end_pts[:, 2]
        
        # Ground/contact and lift thresholds (with hysteresis) and hover band (meters)
        z_g_enter, z_g_exit = 0.06, 0.07
        z_hover = 0.03
        z_lift_enter, z_lift_exit = 0.10, 0.09
        
        # Contact mask with hysteresis
        if not hasattr(self, 'ground_state'):
            self.ground_state = np.zeros_like(end_pts_z, dtype=bool)
        prev_ground_state = self.ground_state
        ground_mask = np.where(prev_ground_state,
                               ~(end_pts_z >= z_g_exit),
                               (end_pts_z <= z_g_enter))
        self.ground_state = ground_mask
        num_ground = int(np.sum(ground_mask))
        
        # Lift mask with hysteresis
        if not hasattr(self, 'lift_state'):
            self.lift_state = np.zeros_like(end_pts_z, dtype=bool)
        prev_lift_state = self.lift_state
        lifted_mask = np.where(prev_lift_state,
                               ~(end_pts_z <= z_lift_exit),
                               (end_pts_z >= z_lift_enter))
        self.lift_state = lifted_mask
        
        # 1) Contact count penalty outside widened [2,6] band (adjusted for single robot: [1,4])
        d_below = max(0, 1 - num_ground)  # CHANGE: adjusted for single robot (6 endpoints vs 12)
        d_above = max(0, num_ground - 4)  # CHANGE: adjusted for single robot
        contact_count_penalty = -0.25 * (d_below + d_above)
        contact_count_penalty = max(contact_count_penalty, -2.0)
        
        # 2) Penalize hover/skating band
        hover_mask = (end_pts_z > 0.0) & (end_pts_z < z_hover)
        hover_count = int(np.sum(hover_mask))
        hover_cap = 3  # CHANGE: adjusted for single robot (was 4)
        hover_term = -float(min(hover_count, hover_cap))
        
        # 3) Reward turnover
        if not hasattr(self, 'prev_ground_mask'):
            self.prev_ground_mask = ground_mask.copy()
        turnover = np.sum(ground_mask ^ self.prev_ground_mask)
        self.prev_ground_mask = ground_mask.copy()
        
        # 4) Per-endpoint duty cycle toward ~50%
        if not hasattr(self, 'ground_duty_ema'):
            self.ground_duty_ema = np.zeros_like(end_pts_z, dtype=np.float32)
        ema_alpha = 0.05
        self.ground_duty_ema = (1.0 - ema_alpha) * self.ground_duty_ema + ema_alpha * ground_mask.astype(np.float32)
        duty_score = -np.sum((self.ground_duty_ema - 0.5) ** 2)
        
        # 4b) Per-endpoint lifted duty cycle
        if not hasattr(self, 'lifted_duty_ema'):
            self.lifted_duty_ema = np.zeros_like(end_pts_z, dtype=np.float32)
        self.lifted_duty_ema = (1.0 - ema_alpha) * self.lifted_duty_ema + ema_alpha * lifted_mask.astype(np.float32)
        lifted_duty_score = -np.sum((self.lifted_duty_ema - 0.5) ** 2)
        
        # 5) Lift transitions bonus
        if not hasattr(self, 'prev_lifted_mask'):
            self.prev_lifted_mask = lifted_mask.copy()
        lift_ups = np.sum(np.logical_and(lifted_mask, ~self.prev_lifted_mask))
        lift_downs = np.sum(np.logical_and(~lifted_mask, self.prev_lifted_mask))
        self.prev_lifted_mask = lifted_mask.copy()
        lift_transition_reward = lift_ups
        
        # Combine endpoint height components
        endpoint_height_reward = (
            1.0 * contact_count_penalty +
            6.0 * turnover +
            0.5 * duty_score +
            1.0 * lifted_duty_score +
            6.0 * lift_transition_reward
        )
        endpoint_height_reward = float(np.clip(endpoint_height_reward, -50.0, 50.0))
        
        # Lifted centroid XY movement
        lifted_centroid_xy_reward = 0.0
        try:
            if np.any(lifted_mask):
                lifted_xy = end_pts[lifted_mask][:, :2]
                centroid_xy = lifted_xy.mean(axis=0)
                if hasattr(self, 'prev_lifted_centroid_xy') and (self.prev_lifted_centroid_xy is not None):
                    centroid_disp = float(np.linalg.norm(centroid_xy - self.prev_lifted_centroid_xy))
                    lifted_centroid_xy_reward = centroid_disp
                self.prev_lifted_centroid_xy = centroid_xy.copy()
            else:
                self.prev_lifted_centroid_xy = None
        except Exception:
            lifted_centroid_xy_reward = 0.0
        
        # Continuous movement: per-step COM XY progress
        com_step_progress = 0.0
        com_step_dist = 0.0
        com_step_direction = None
        if hasattr(self, 'prev_pos') and self.prev_pos is not None:
            com_step_vec = robot_pos[:2] - self.prev_pos[:2]
            com_step_dist = float(np.linalg.norm(com_step_vec))
            com_step_progress = float(np.tanh(5.0 * com_step_dist))
            if com_step_dist > 1e-6:
                com_step_direction = com_step_vec / com_step_dist
            else:
                com_step_direction = None
        else:
            com_step_direction = None

        # --- Direction Consistency Streak Reward ---
        # Track the direction of COM movement and reward streaks of consistent direction
        direction_streak_reward = 0.0
        direction_streak_weight = 10.0  # Tune this weight as needed
        direction_cos_threshold = 0.92  # ~23 degrees, cos(angle)
        direction_decay = 0.5  # How much to decay streak on inconsistency
        if not hasattr(self, 'direction_streak'):
            self.direction_streak = 0
            self.prev_direction = None
        if com_step_direction is not None:
            if self.prev_direction is not None:
                cos_sim = float(np.dot(com_step_direction, self.prev_direction))
                if cos_sim > direction_cos_threshold:
                    self.direction_streak += 1
                else:
                    # Partial decay, not full reset
                    self.direction_streak = int(self.direction_streak * direction_decay)
            else:
                self.direction_streak = 1
            self.prev_direction = com_step_direction.copy()
        else:
            # No movement, decay streak
            self.direction_streak = int(self.direction_streak * direction_decay)
        # Reward is proportional to streak length (with tanh squashing)
        direction_streak_reward = direction_streak_weight * float(np.tanh(0.1 * self.direction_streak))
        
        # Grounded centroid XY movement
        grounded_centroid_xy_reward = 0.0
        try:
            if np.any(ground_mask):
                grounded_xy = end_pts[ground_mask][:, :2]
                g_centroid = grounded_xy.mean(axis=0)
                if hasattr(self, 'prev_grounded_centroid_xy') and (self.prev_grounded_centroid_xy is not None):
                    g_disp = float(np.linalg.norm(g_centroid - self.prev_grounded_centroid_xy))
                    grounded_centroid_xy_reward = g_disp
                self.prev_grounded_centroid_xy = g_centroid.copy()
            else:
                self.prev_grounded_centroid_xy = None
        except Exception:
            grounded_centroid_xy_reward = 0.0
        
        # Stagnation penalty
        if not hasattr(self, 'low_speed_hist'):
            self.low_speed_hist = []
        xy_speed_for_stall = (com_step_dist / self.dt) if (hasattr(self, 'prev_pos') and self.prev_pos is not None) else 0.0
        self.low_speed_hist.append(xy_speed_for_stall < 0.01)
        if len(self.low_speed_hist) > 15:
            self.low_speed_hist.pop(0)
        stall_ratio = float(np.mean(self.low_speed_hist)) if len(self.low_speed_hist) > 0 else 0.0
        
        if not hasattr(self, 'stall_streak'):
            self.stall_streak = 0
        self.stall_streak = self.stall_streak + 1 if (xy_speed_for_stall < 0.01) else 0
        ramp = float(1.0 - np.exp(-0.1 * self.stall_streak))
        stall_penalty = -4.0 * stall_ratio * (0.5 + 0.5 * ramp)
        
        if (com_step_progress >= 0.02) or (turnover > 0) or (lift_transition_reward > 0):
            self.stall_streak = 0
            ramp = 0.0
        stall_penalty = max(stall_penalty, -0.5)
        
        # Resume bonus
        if not hasattr(self, 'high_speed_hist'):
            self.high_speed_hist = []
        self.high_speed_hist.append(xy_speed_for_stall >= 0.02)
        if len(self.high_speed_hist) > 5:
            self.high_speed_hist.pop(0)
        resume_bonus = 0.0
        if len(self.high_speed_hist) >= 3 and all(self.high_speed_hist[-3:]):
            resume_bonus = 2.0
        
        # Track lifted streaks
        if not hasattr(self, 'lift_streaks'):
            self.lift_streaks = np.zeros_like(end_pts_z, dtype=np.int32)
        self.lift_streaks = np.where(lifted_mask, self.lift_streaks + 1, 0)
        
        # Lifted swing path-length
        if not hasattr(self, 'prev_end_pts'):
            self.prev_end_pts = end_pts.copy()
        step_xy_all = np.linalg.norm(end_pts[:, :2] - self.prev_end_pts[:, :2], axis=1)
        warmup, alpha_decay = 6, 0.05
        dwell_decay = np.exp(-alpha_decay * np.maximum(0, self.lift_streaks - warmup)).astype(np.float32)
        lifted_step_lengths = step_xy_all * lifted_mask.astype(np.float32) * dwell_decay
        lifted_swing_reward = float(np.sum(np.tanh(5.0 * lifted_step_lengths)))
        
        dwell_thresh = 12
        lift_dwell_penalty = -0.02 * float(np.sum(np.maximum(0, self.lift_streaks - dwell_thresh)))
        lift_dwell_penalty = max(lift_dwell_penalty, -1.0)
        self.prev_end_pts = end_pts.copy()
        
        # Hover penalty gating
        lifted_count = int(np.sum(lifted_mask))
        hover_weight = 0.0 if (self.stall_streak < 5 or lifted_count >= 2) else 3.0
        
        # Action smoothing penalty
        action_smooth_penalty = 0.0
        if controls is not None:
            if hasattr(self, 'prev_controls') and self.prev_controls is not None:
                try:
                    action_smooth_penalty = -0.5 * float(np.sum(np.abs(controls - self.prev_controls)))
                except Exception:
                    action_smooth_penalty = 0.0
            self.prev_controls = controls.copy()
        action_smooth_penalty = max(action_smooth_penalty, -4.0)

        # Reward rotation of IMU about the x-axis of the IMU
        # IMU rotation reward: encourage rotation about each rod's local x-axis
        try:
            imu_vecs = self._get_IMU_gravity_vectors().reshape(self.n_rods, 3)  # (n_rods, 3)
            # Angle around local x-axis: rotation of gravity shows up in y (sin) and z (-cos)
            angles_x = np.arctan2(imu_vecs[:, 1], -imu_vecs[:, 2])  # radians

            # Initialize previous angles if missing
            if not hasattr(self, 'prev_imu_angles') or self.prev_imu_angles is None:
                self.prev_imu_angles = angles_x.copy()

            # Angle difference with wrapping to [-pi, pi]
            d_angles = angles_x - self.prev_imu_angles
            d_angles = (d_angles + np.pi) % (2.0 * np.pi) - np.pi

            # Angular speed (rad/s)
            ang_speed = d_angles / max(self.dt, 1e-8)

            # Reward proportional to absolute angular speed, squashed with tanh to keep bounded
            imu_rotation_reward = float(np.sum(np.tanh(0.5 * np.abs(ang_speed))))

            # Update stored angles for next step
            self.prev_imu_angles = angles_x.copy()
        except Exception as e:
            # On any failure, set zero reward and avoid breaking simulation
            imu_rotation_reward = 0.0
            if not hasattr(self, 'prev_imu_angles'):
                self.prev_imu_angles = None

        # Compose final reward
        lifted_centroid_xy_reward_weight = 20.0
        grounded_centroid_xy_reward_weight = 5.0
        imu_rotation_reward_weight = 10.0
        
        reward_raw = (
            endpoint_height_reward
            + lifted_centroid_xy_reward_weight * lifted_centroid_xy_reward
            + grounded_centroid_xy_reward_weight * grounded_centroid_xy_reward
            + 15.0 * com_step_progress  # Increased from 24.0 to 30.0
            + 10.0 * lifted_swing_reward  # Increased from 7.0 to 10.0
            + stall_penalty
            + resume_bonus
            + lift_dwell_penalty
            + hover_weight * hover_term
            + action_smooth_penalty
            + direction_streak_reward
            + imu_rotation_reward_weight * imu_rotation_reward
        )
        
        # Final reward clipping
        reward = float(np.clip(reward_raw, -200.0, 200.0))
        
        # Update position tracking
        self.prev_pos = robot_pos.copy()
        self.step_count += 1
        
        # Build observation
        observation = self.get_observation()
        done = False
        
        # Detailed info for debugging
        info = {
            'endpoint_height_reward': endpoint_height_reward,
            'contact_count_penalty': float(contact_count_penalty),
            'turnover': float(turnover),
            'duty_score': float(duty_score),
            'lifted_duty_score': float(lifted_duty_score),
            'hover_count': float(hover_count),
            'hover_term': float(hover_term),
            'lift_transition_reward': float(lift_transition_reward),
            'lifted_centroid_xy_reward': float(lifted_centroid_xy_reward),
            'grounded_centroid_xy_reward': float(grounded_centroid_xy_reward),
            'com_step_progress': float(com_step_progress),
            'lifted_swing_reward': float(lifted_swing_reward),
            'stall_penalty': float(stall_penalty),
            'resume_bonus': float(resume_bonus),
            'lift_dwell_penalty': float(lift_dwell_penalty),
            'lift_streak_mean': float(np.mean(self.lift_streaks)),
            'hover_weight_applied': float(hover_weight),
            'action_smooth_penalty': float(action_smooth_penalty),
            'direction_streak_reward': float(direction_streak_reward),
            'direction_streak': int(self.direction_streak),
            'controls': controls.copy() if controls is not None else None,
            'action': self.prev_action.copy(),
        }
        
        return observation, reward, done, info

    def get_observation(self):
        """Build observation vector from configured components."""
        parts = []
        
        # 1. Cable lengths (normalized)
        if self.obs_components.get('cable_lengths', 0) > 0:
            lengths = self._get_actuated_cable_lengths()
            norm_lengths = (lengths - self.min_cable_length) / max(self.max_cable_length - self.min_cable_length, 1e-6)
            norm_lengths = np.clip(norm_lengths, 0.0, 1.0).astype(np.float32)
            parts.append(norm_lengths)
        
        # 2. Cable length rates
        if self.obs_components.get('cable_rates', 0) > 0:
            curr_lengths = self._get_actuated_cable_lengths()
            rates = (curr_lengths - self.prev_lengths) / max(self.dt, 1e-8)
            denom = max(self.max_cable_length - self.min_cable_length, 1e-6)
            rates_norm = np.clip(rates / denom, -1.0, 1.0).astype(np.float32)
            parts.append(rates_norm)
            self.prev_lengths = curr_lengths
        
        # 3. Previous action
        if self.obs_components.get('prev_action', 0) > 0:
            parts.append(self.prev_action.astype(np.float32))
        
        # 4. IMU gravity vectors
        if self.obs_components.get('imu', 0) > 0:
            imu_grav = self._get_IMU_gravity_vectors()
            parts.append(imu_grav)
        
        # Concatenate all parts
        obs = np.concatenate(parts).astype(np.float32) if parts else np.zeros(1, dtype=np.float32)
        
        # Verify dimension matches
        if obs.shape[0] != self.obs_dim:
            debug_print(f"WARNING: Observation dimension mismatch! Expected {self.obs_dim}, got {obs.shape[0]}", 
                       "single_tensegrity_mjc_simulation.py", True)
        
        return obs

    def _get_actuated_cable_lengths(self) -> np.ndarray:
        """Get current lengths of actuated cables."""
        lengths = []
        for i in range(self.n_actuators):
            try:
                s0 = self.mjc_data.sensor(f"pos_{self.cable_sites[i][0]}").data
                s1 = self.mjc_data.sensor(f"pos_{self.cable_sites[i][1]}").data
                length = np.linalg.norm(s1 - s0)
                lengths.append(length)
            except Exception as e:
                debug_print(f"Error getting cable {i} length: {e}", "single_tensegrity_mjc_simulation.py", self.debug_enabled)
                lengths.append(0.0)
        return np.array(lengths, dtype=np.float32)

    def _get_IMU_gravity_vectors(self) -> np.ndarray:
        """Get gravity vectors in rod body frames (3 rods × 3D = 9 values)."""
        vecs = []
        gravity_world = np.array([0, 0, -1.0], dtype=np.float32)
        
        for geom_name in self.imu_geom_names:
            try:
                # Get body associated with this geom
                geom_id = mujoco.mj_name2id(self.mjc_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                body_id = self.mjc_model.geom_bodyid[geom_id]
                quat = self.mjc_data.xquat[body_id]
                
                # Convert quaternion to rotation matrix
                w, x, y, z = quat
                R = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                    [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                    [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
                ], dtype=np.float32)
                
                # Transform gravity to body frame
                grav_body = R.T @ gravity_world
                vecs.append(grav_body.astype(np.float32))
            except Exception as e:
                debug_print(f"Error getting IMU for {geom_name}: {e}", "single_tensegrity_mjc_simulation.py", self.debug_enabled)
                vecs.append(np.zeros(3, dtype=np.float32))
        
        return np.concatenate(vecs).astype(np.float32)

    def get_endpts(self):
        """Get endpoint xyz coordinates."""
        end_pts = []
        for end_pt_site in self.end_pts:
            try:
                end_pt = self.mjc_data.sensor(f"pos_{end_pt_site}").data
                end_pts.append(end_pt)
            except Exception as e:
                debug_print(f"Error getting endpoint {end_pt_site}: {e}", "single_tensegrity_mjc_simulation.py", self.debug_enabled)
                end_pts.append(np.zeros(3, dtype=np.float32))
        
        return np.vstack(end_pts)

    def get_robot_position(self):
        """Returns the current position of the robot (mean of endpoints)."""
        self.forward()
        end_pts = self.get_endpts()
        return end_pts.mean(axis=0)

    def render(self, mode='human', **kwargs):
        """Render the simulation in real-time viewer.
        
        Parameters
        ----------
        mode : str, default='human'
            Rendering mode ('human' for window, 'rgb_array' for frame)
        **kwargs : dict
            Additional rendering arguments
        
        Returns
        -------
        None or np.ndarray
            None for human mode, frame array for rgb_array mode
        """
        if not self.visualize:
            return None

        # Lazy-load viewer
        if self._viewer is None:
            try:
                from mujoco import viewer
                self._viewer = viewer.launch_passive(self.mjc_model, self.mjc_data)
            except Exception as e:
                print(f"[ERROR] Failed to launch viewer: {e}")
                return None

        try:
            # Update camera to track robot
            end_pts = self.get_endpts()
            robot_pos = end_pts.mean(axis=0)
            self._viewer.cam.lookat[:] = robot_pos
            self._viewer.cam.distance = 12.0  # Closer for single robot
            self._viewer.cam.elevation = -25
            self._viewer.cam.azimuth = 90

            # Update scene
            self._viewer.sync()
            self._viewer.render()
        except Exception as e:
            pass
            # Silently fail to avoid spam

    def close(self):
        """Close renderer and cleanup resources."""
        try:
            if hasattr(self, '_viewer') and self._viewer is not None:
                self._viewer.close()
            if hasattr(self, 'renderer') and self.renderer is not None:
                self.renderer.close()
            import cv2
            cv2.destroyAllWindows()
        except:
            pass


__all__ = ["SingleTensegrityMuJoCoSimulator"]
