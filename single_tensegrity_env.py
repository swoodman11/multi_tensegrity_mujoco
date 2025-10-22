"""Gymnasium environment wrapper for single tensegrity robot.

This environment wraps SingleTensegrityMuJoCoSimulator to provide a standard
Gymnasium interface for reinforcement learning with Stable-Baselines3.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path

# Import single tensegrity simulator
from mujoco_physics_engine.single_tensegrity_mjc_simulation import SingleTensegrityMuJoCoSimulator


def debug_print(message, filename="single_tensegrity_env.py", debug_enabled=False):
    """Print debug messages with filename prefix if debug is enabled"""
    if debug_enabled:
        print(f"DEBUG {filename}: {message}")


class SingleTensegrityEnv(gym.Env):
    """Gymnasium environment for single 3-bar tensegrity robot.
    
    Action space: Box(0, 1, (6,)) - normalized cable target lengths
    Observation space: Box(-inf, inf, (obs_dim,)) - robot state (default 27-dimensional)
    """
    
    def __init__(self, obs_dim=None, visualize=False, debug_enabled=False, max_episode_steps=50, render_pause=0.01):
        """Initialize single tensegrity environment.
        
        Parameters
        ----------
        obs_dim : int, optional
            Observation dimension. If None, calculated automatically (default 27).
        visualize : bool, default=False
            Whether to enable visualization.
        debug_enabled : bool, default=False
            Whether to enable debug printing.
        max_episode_steps : int, default=50
            Maximum number of steps per episode before truncation.
            With 100 physics timesteps per action (1 second), 50 steps = 50 seconds.
        render_pause : float, default=0.01
            Pause duration between renders in seconds (for visualization speed control).
        """
        super().__init__()
        
        self.debug_enabled = debug_enabled
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        
        # Setup the simulator with Path object
        xml_path = Path("mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml")
        self.sim = SingleTensegrityMuJoCoSimulator(
            xml_path=xml_path,
            obs_dim=obs_dim,
            visualize=visualize,
            render_size=(800, 600),
            render_fps=30,
            debug_enabled=debug_enabled,
            controller_kp=10.0,
            controller_ki=0.2,
            controller_kd=2.0,
            render_pause=render_pause
        )
        
        debug_print(f"SingleTensegrityMuJoCoSimulator initialized with visualize={visualize}", 
                   "single_tensegrity_env.py", debug_enabled)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.sim.obs_dim,), 
            dtype=np.float32
        )
        
        # Define action space - 6 actuated cables
        self.num_actuators = 6
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.num_actuators,), 
            dtype=np.float32
        )
        
        debug_print(f"Environment initialized:", "single_tensegrity_env.py", debug_enabled)
        debug_print(f"  Observation space: {self.observation_space.shape}", "single_tensegrity_env.py", debug_enabled)
        debug_print(f"  Action space: {self.action_space.shape}", "single_tensegrity_env.py", debug_enabled)
        debug_print(f"  Simulator obs_dim: {self.sim.obs_dim}", "single_tensegrity_env.py", debug_enabled)
        debug_print(f"  Simulator n_actuators: {self.sim.n_actuators}", "single_tensegrity_env.py", debug_enabled)
        debug_print(f"  Max episode steps: {self.max_episode_steps}", "single_tensegrity_env.py", debug_enabled)
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional options (unused).
        
        Returns
        -------
        observation : np.ndarray
            Initial observation.
        info : dict
            Additional information.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset step counter
        self._elapsed_steps = 0
        
        obs = self.sim.reset()
        info = {}  # Required by new Gymnasium API
        
        debug_print(f"Environment reset, obs shape: {obs.shape}", "single_tensegrity_env.py", self.debug_enabled)
        
        return obs, info
    
    def step(self, action):
        """Execute one environment step.
        
        Parameters
        ----------
        action : np.ndarray, shape (6,)
            Normalized target cable lengths in [0, 1].
        
        Returns
        -------
        observation : np.ndarray
            Current observation.
        reward : float
            Reward for this step.
        terminated : bool
            Whether episode is terminated.
        truncated : bool
            Whether episode is truncated (time limit).
        info : dict
            Additional information about the step.
        """
        # Simulate one timestep by passing target lengths to simulator
        obs, reward, done, info = self.sim.sim_step(action)
        
        # Increment step counter
        self._elapsed_steps += 1
        
        # Handle truncation - episode ends after max_episode_steps (500 steps)
        truncated = self._elapsed_steps >= self.max_episode_steps
        
        debug_print(f"Step {self._elapsed_steps}/{self.max_episode_steps}: obs_shape={obs.shape}, reward={reward:.3f}, truncated={truncated}", 
                   "single_tensegrity_env.py", self.debug_enabled)
        
        return obs, reward, done, truncated, info
    
    def render(self, mode='human'):
        """Render the environment.
        
        Parameters
        ----------
        mode : str, default='human'
            Rendering mode.
        
        Returns
        -------
        frame : np.ndarray or None
            Rendered frame if mode is 'rgb_array', None otherwise.
        """
        try:
            return self.sim.render(mode)
        except Exception as e:
            debug_print(f"Render failed: {e}", "single_tensegrity_env.py", self.debug_enabled)
            return None
    
    def close(self):
        """Close the environment and cleanup resources."""
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass
        
        debug_print("Environment closed", "single_tensegrity_env.py", self.debug_enabled)


__all__ = ["SingleTensegrityEnv"]
