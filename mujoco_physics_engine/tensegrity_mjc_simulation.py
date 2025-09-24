import multiprocessing
from pathlib import Path
from typing import List
from PIL import Image

import mujoco
import numpy as np

from mujoco_physics_engine.cable_motor import DCMotor
from mujoco_physics_engine.mujoco_simulation import AbstractMuJoCoSimulator
from mujoco_physics_engine.pid import PID


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
                 obs_dim: int = 78):
        super().__init__(xml_path, visualize, render_size, render_fps)
        self.min_cable_length = 0.6
        self.max_cable_length = 2.4
        self.n_actuators = num_actuated_cables
        self.curr_ctrl = [0.0 for _ in range(num_actuated_cables)]
        self.pids = [PID() for _ in range(num_actuated_cables)]
        self.cable_motors = [DCMotor() for _ in range(num_actuated_cables)]
        self.n_rods = num_rods
        self.n_cables = self.mjc_model.tendon_stiffness.shape[0]
        self.actuated_ids = (list(range(num_actuated_cables // 2))
                             + list(range(self.n_cables // 2, self.n_cables // 2 + num_actuated_cables // 2)))
        self.obs_dim = obs_dim  # Dimension of observation space

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

    def bring_to_grnd(self):
        """
        Finds the z-translation that would bring the lowest end cap to the ground, and aplies it to the robot
        """
        self.forward()
        qpos = self.mjc_data.qpos.copy().reshape(-1, 7)
        end_pts = self.get_endpts().reshape(-1, 3)
        min_z = end_pts[:, 2].min()
        qpos[:, 2] -= min_z - 0.175
        self.mjc_data.qpos = qpos.reshape(1, -1)

    def reset(self):
        """
        Resets the robots as if it was just instantiated from the xml file.
        """
        super().reset()
        self.bring_to_grnd()

        for motor in self.cable_motors:
            motor.reset_omega_t()
        
        self.prev_pos = None
        self.step_count = 0
        
        # Return observation for RL (ADD THIS LINE):
        obs = self.get_endpts().flatten()
        return obs[:self.obs_dim] if hasattr(self, 'obs_dim') else obs[:78]

    def reset_actuators(self):
        for motor in self.cable_motors:
            motor.reset_omega_t()

        for pid in self.pids:
            pid.reset()


    def sim_step(self, controls=None):
        """
        Takes a single simulation step given controls. Controls must be list of np.array that matches the n_actuators
        """
        ctrl_idx = 0

        self.forward()
        for i in range(len(self.cable_sites)):
            sites = self.cable_sites[i]
            rest_length = self.mjc_model.tendon_lengthspring[i, 0]

            # HACK to have tendons only apply tension but not compression forces
            s0 = self.mjc_data.sensor(f"pos_{sites[0]}").data
            s1 = self.mjc_data.sensor(f"pos_{sites[1]}").data
            dist = np.linalg.norm(s1 - s0)
            self.mjc_model.tendon_stiffness[i] = 0 if dist < rest_length else self.stiffness[i]

            if controls is not None and i in self.actuated_ids:
                ctrl = np.array(controls[ctrl_idx])

                # Compute change in cable rest lengths basted on ctrl
                dl = self.cable_motors[ctrl_idx].compute_cable_length_delta(ctrl, self.dt)
                rest_length = rest_length - dl
                self.mjc_model.tendon_lengthspring[self.actuated_ids[ctrl_idx]] = rest_length

                ctrl_idx += 1

        mujoco.mj_step(self.mjc_model, self.mjc_data)
        self.forward()

        # Get end points for locomotion reward
        end_pts = self.get_endpts()
        robot_pos = end_pts.mean(axis=0)  # Use the mean of end points as the robot's position
        
        # Calculate forward motion reward
        if hasattr(self, 'prev_pos') and self.prev_pos is not None:
            forward_velocity = (robot_pos[0] - self.prev_pos[0]) / self.dt
            reward = forward_velocity * 10.0  # Reward forward motion
        else:
            reward = 0.0
        
        self.prev_pos = robot_pos.copy()  # Use .copy() to avoid reference issues
        
        # ADD step counter:
        self.step_count = getattr(self, 'step_count', 0) + 1


        # Construct observation
        # qpos = self.mjc_data.qpos.copy().reshape(-1, 7)
        # qvel = self.mjc_data.qvel.copy().reshape(-1, 6)
        # observation = np.concatenate([
        #     qpos.flatten(),
        #     qvel.flatten(),
        #     end_pts.flatten()
        # ])
        # observation = observation[:self.obs_dim]  # Ensure observation matches obs_dim
        observation = self.get_endpts().flatten()
        if hasattr(self, 'obs_dim'):
            observation = observation[:self.obs_dim]
        else:
            observation = observation[:78]
        
        # Remove any NaN values:
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        done = False
        info = {}
        
        return observation, reward, done, info

    def get_robot_position(self):
        """
        Returns the current position of the robot.
        """
        self.forward()
        print("Tensegrity positions: ", self.mjc_data.qpos)
        return self.mjc_data.qpos[:3]  # Assuming the first three elements represent the robot's position

    def get_endpts(self):
        # Get end point xyz coordinates
        end_pts = []
        for end_pt_site in self.end_pts:
            end_pt = self.mjc_data.sensor(f"pos_{end_pt_site}").data
            end_pts.append(end_pt)

        end_pts = np.vstack(end_pts)
        return end_pts
    
    def render(self, mode='human', width=800, height=600):
        """Render the simulation"""
        try:
            # Import here to avoid issues
            import mujoco
            
            # Create renderer if it doesn't exist
            if not hasattr(self, '_renderer'):
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
            print(f"Rendering failed: {e}")
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


    def run_target_lengths(self, target_lengths, max_gait_time=6.0, vis_save_dir: Path = None, vis_prefix: str = ""):
        self.reset_actuators()

        step = 0
        max_steps = int(max_gait_time / self.dt)
        controls = [1.0]  # dummy value for init
        frames = []

        while any([c != 0 for c in controls]) and step < max_steps:
            step += 1
            curr_lengths = []
            self.forward()

            controls = []
            for i in range(len(target_lengths)):
                pid = self.pids[i]
                lengths = target_lengths[i]

                rest_length = self.mjc_model.tendon_lengthspring[i, 0]

                idx = self.cable_map[i] if hasattr(self, "cable_map") and self.cable_map else self.actuated_ids[i]
                s0 = self.mjc_data.sensor(f"pos_{self.cable_sites[idx][0]}").data
                s1 = self.mjc_data.sensor(f"pos_{self.cable_sites[idx][1]}").data
                curr_length = np.linalg.norm(s1 - s0)
                curr_lengths.append(curr_length)

                ctrl, _ = pid.update_control_by_target_norm_length(curr_length, lengths, rest_length)
                controls.append(ctrl)

            self.sim_step(controls)

            if vis_save_dir is not None:
                if not vis_save_dir.exists():
                    vis_save_dir.mkdir(exist_ok=True)
                frame = self.render_frame('front')
                frames.append(frame)
                Image.fromarray(frame).save(vis_save_dir / f"{vis_prefix}_{step}.png")

        return frames

class MultiProcTensegrityMujocoSimulator:

    def __init__(self,
                 num_sims,
                 xml_path: Path,
                 # visualize: bool = True,
                 render_size: (int, int) = (240, 240),
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
    # Add this method to your TensegrityMuJoCoSimulator class (at the end of the class)
    
