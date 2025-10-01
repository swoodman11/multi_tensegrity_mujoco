import gymnasium as gym
from gymnasium  import spaces
import numpy as np
from pathlib import Path

# Import your simulator
from mujoco_physics_engine.tensegrity_mjc_simulation import TensegrityMuJoCoSimulator as Simulator # Adjust import if needed

def debug_print(message, filename="tensegrity_env.py", debug_enabled=False):
    """Print debug messages with filename prefix if debug is enabled"""
    if debug_enabled:
        print(f"DEBUG {filename}: {message}")

class TensegrityEnv(gym.Env):
    def __init__(self, obs_dim=None, visualize=False, obs_mode: str = "tier2", debug_enabled=False):
        super().__init__()
        
        self.debug_enabled = debug_enabled
        
        # Setup the simulator with Path object
        xml_path = Path("mujoco_physics_engine/xml_models/two_3bar_new_platform_config_1.xml")
        self.sim = Simulator(
            xml_path=xml_path,
            obs_dim=obs_dim,
            obs_mode=obs_mode,
            visualize=visualize,
            render_size=(800, 600),
            render_fps=30,
            debug_enabled=debug_enabled
        )
        
        debug_print(f"Simulator.__init__ called with visualize={visualize}", "tensegrity_env.py", debug_enabled)

        # Define observation spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sim.obs_dim,), dtype=np.float32)
        
        # Define action spaces
        self.num_actuators = 12  # 12 actuated cables
        # Action space consisting of normalized cable target lengths
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_actuators,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.sim.reset()
        info = {}  # Required by new Gym API
        return obs, info

    def step(self, action):
        # Simulate one timestep by passing in target lengths to the simulator
        obs, reward, done, info = self.sim.sim_step(action)

        # Handle truncation (new in Gymnasium)
        truncated = False  # Set to True if episode ends due to time limit

        return obs, reward, done, truncated, info

    # Add this method to your TensegrityMuJoCoSimulator class (at the end of the class)
    def render(self, mode='human'):
        """Render the environment"""
        try:
            return self.sim.render(mode)  # Call simulator's render, not self
        except Exception as e:
            debug_print(f"Render failed: {e}", "tensegrity_env.py", self.debug_enabled)
            return None

    def close(self):
        """Close the environment"""
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass