import gymnasium as gym
from gymnasium  import spaces
import numpy as np

# Import your simulator
from mujoco_physics_engine.tensegrity_mjc_simulation import TensegrityMuJoCoSimulator as Simulator # Adjust import if needed

class TensegrityEnv(gym.Env):
    def __init__(self, visualize=False):
        super().__init__()
        
        # Setup the simulator
        self.sim = Simulator(xml_path="mujoco_physics_engine/xml_models/two_3bar_new_platform_config_1.xml",obs_dim=36,
            visualize=visualize,  # Use the parameter
            render_size=(800, 600),
            render_fps=30)  # Adjust to match your setup
        self.sim.reset()
        
        print(f"DEBUG: Simulator.__init__ called with visualize={visualize}")

        # Define action and observation spaces
        self.num_actuators = 12  # Example: 12 muscle cables
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sim.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actuators,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.sim.reset()
        info = {}  # Required by new Gym API
        return obs, info

    def step(self, action):
        # Simulate one timestep
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
            print(f"Render failed: {e}")
            return None

    def close(self):
        """Close the environment"""
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass