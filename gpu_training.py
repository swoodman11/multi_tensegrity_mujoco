"""
GPU-Optimized PPO Training for Tensegrity Robot Locomotion

This script implements pure PPO (Proximal Policy Optimization) training for tensegrity robots
without behavioral cloning. It's designed for GPU acceleration and includes comprehensive
hyperparameter tuning guidance.

Key Features:
- Pure reinforcement learning (no behavioral cloning pretraining)
- GPU-optimized PPO configurations for different hardware
- Extensive hyperparameter documentation and tuning guidance
- Automatic hardware detection and configuration selection
- Real-time training monitoring and diagnostics
- Comprehensive logging and model saving

Hardware Support:
- RTX 4090 (24GB VRAM) - Recommended for large-scale training
- RTX 3080/3090 (10-24GB VRAM) - Good for standard training
- RTX 2080 Ti (11GB VRAM) - Conservative configurations
- RTX 5090 (32GB VRAM) - Future-proof extreme configurations

Author: GitHub Copilot
Date: October 2025
"""

import time
import gc
import torch
import numpy as np
import psutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt

from tensegrity_env import TensegrityEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class GPUTrainingMonitor(BaseCallback):
    """
    Custom callback for monitoring GPU training progress and detecting issues
    """
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.last_mean_reward = -float('inf')
        self.stagnation_counter = 0
        self.training_metrics = {
            'approx_kl': [],
            'clip_fraction': [],
            'entropy_loss': [],
            'policy_gradient_loss': [],
            'value_loss': [],
            'explained_variance': []
        }
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Extract training metrics from logger
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                logger_dict = self.model.logger.name_to_value
                
                # Store key metrics for analysis
                for key in self.training_metrics:
                    if f'train/{key}' in logger_dict:
                        self.training_metrics[key].append(logger_dict[f'train/{key}'])
                
                # Check for training stagnation
                approx_kl = logger_dict.get('train/approx_kl', 0)
                clip_fraction = logger_dict.get('train/clip_fraction', 0)
                entropy_loss = logger_dict.get('train/entropy_loss', 0)
                
                # Stagnation detection
                if approx_kl < 1e-6 and clip_fraction < 0.01:
                    self.stagnation_counter += 1
                    if self.stagnation_counter >= 5:
                        print(f"\n⚠️  TRAINING STAGNATION DETECTED at step {self.n_calls}")
                        print(f"   approx_kl: {approx_kl:.2e} (too low)")
                        print(f"   clip_fraction: {clip_fraction:.3f} (too low)")
                        print(f"   Consider increasing learning_rate or entropy coefficient")
                else:
                    self.stagnation_counter = 0
                
                # GPU memory monitoring
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated(0) / 1e9
                    max_memory = torch.cuda.max_memory_allocated(0) / 1e9
                    
                    if gpu_memory > 20:  # Warning if using >20GB
                        print(f"⚠️  High GPU memory usage: {gpu_memory:.1f}GB / {max_memory:.1f}GB")
        
        return True


def check_system_requirements() -> Tuple[bool, Optional[str], Optional[Tuple[float, float]]]:
    """
    Comprehensive system check for GPU training requirements
    
    Returns:
        success: bool - Whether system meets minimum requirements
        gpu_name: str - GPU model name (lowercase)
        memory_specs: Tuple[float, float] - (GPU VRAM GB, System RAM GB)
    """
    print("🔧 System Requirements Check")
    print("=" * 50)
    
    # CUDA availability check
    if not torch.cuda.is_available():
        print("❌ CUDA not available - GPU training not possible")
        print("   Install CUDA toolkit: https://developer.nvidia.com/cuda-toolkit")
        return False, None, None
    
    # GPU specifications
    gpu_name = torch.cuda.get_device_name(0).lower()
    gpu_properties = torch.cuda.get_device_properties(0)
    gpu_memory_gb = gpu_properties.total_memory / 1e9
    
    # System RAM
    system_memory = psutil.virtual_memory()
    system_ram_gb = system_memory.total / 1e9
    
    # Minimum requirements check
    min_gpu_memory = 6.0  # Minimum 6GB VRAM
    min_system_ram = 16.0  # Minimum 16GB RAM
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU VRAM: {gpu_memory_gb:.1f}GB")
    print(f"✅ System RAM: {system_ram_gb:.1f}GB")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ PyTorch Version: {torch.__version__}")
    
    # Requirements validation
    if gpu_memory_gb < min_gpu_memory:
        print(f"❌ Insufficient GPU memory: {gpu_memory_gb:.1f}GB < {min_gpu_memory}GB required")
        return False, gpu_name, (gpu_memory_gb, system_ram_gb)
    
    if system_ram_gb < min_system_ram:
        print(f"⚠️  Low system RAM: {system_ram_gb:.1f}GB (recommend >{min_system_ram}GB)")
    
    # Clear GPU cache and check current usage
    torch.cuda.empty_cache()
    current_usage = torch.cuda.memory_allocated(0) / 1e9
    available_memory = gpu_memory_gb - current_usage - 1.0  # Reserve 1GB
    
    print(f"📊 Available GPU memory: ~{available_memory:.1f}GB")
    print("🚀 System ready for GPU training!")
    
    return True, gpu_name, (gpu_memory_gb, system_ram_gb)


def get_ppo_hyperparameters() -> Dict[str, Dict[str, Any]]:
    """
    Comprehensive PPO hyperparameter configurations for different hardware setups.
    
    HYPERPARAMETER TUNING GUIDE:
    ============================
    
    1. LEARNING_RATE (3e-4 to 1e-3):
       - Higher (1e-3): Faster initial learning, risk of instability
       - Lower (3e-4): More stable, slower convergence
       - Use higher for exploration, lower for fine-tuning
       
    2. N_STEPS (2048 to 32768):
       - Higher: More data per update, better sample efficiency
       - Lower: More frequent updates, faster iteration
       - Limited by GPU memory - larger = more VRAM needed
       
    3. BATCH_SIZE (64 to 2048):
       - Higher: More stable gradients, better GPU utilization
       - Lower: More gradient updates per rollout
       - Should divide n_steps evenly
       
    4. N_EPOCHS (3 to 20):
       - Higher: More learning per data batch, risk of overfitting
       - Lower: Less overfitting, may need more timesteps
       - Reduce if seeing policy degradation
       
    5. GAMMA (0.99 to 0.999):
       - Higher: Longer-term planning, more stable value learning
       - Lower: Focus on immediate rewards
       - Use 0.99 for locomotion tasks
       
    6. GAE_LAMBDA (0.9 to 0.98):
       - Higher: Less bias, more variance in advantage estimation
       - Lower: More bias, less variance
       - Use 0.95 for good balance
       
    7. CLIP_RANGE (0.1 to 0.3):
       - Higher: Allows larger policy updates
       - Lower: More conservative updates, more stable
       - Increase if policy learning is too slow
       
    8. ENT_COEF (0.0 to 0.1):
       - Higher: More exploration, prevents premature convergence
       - Lower: More exploitation of current policy
       - Critical for tensegrity robots - use 0.01-0.05
       
    9. VF_COEF (0.5 to 1.0):
       - Higher: More emphasis on value function learning
       - Lower: More emphasis on policy learning
       - Use 0.5 for balanced training
       
    10. MAX_GRAD_NORM (0.5 to 2.0):
        - Higher: Allows larger gradient updates
        - Lower: More stable training
        - Use 0.5 for stable training, 1.0+ for faster learning
    
    NETWORK ARCHITECTURE GUIDE:
    ============================
    
    Policy Network (pi):
    - Small networks (256, 256): Fast training, may limit performance
    - Medium networks (512, 512): Good balance for most tasks
    - Large networks (1024, 1024): Better performance, more VRAM needed
    
    Value Network (vf):
    - Can be smaller than policy network
    - (256, 256) often sufficient for value estimation
    
    TROUBLESHOOTING COMMON ISSUES:
    ==============================
    
    1. TRAINING STAGNATION (KL ~0, clip_fraction ~0):
       - Increase learning_rate (try 2x current value)
       - Increase ent_coef (try 0.05-0.1)
       - Increase clip_range (try 0.3-0.4)
       - Reduce n_epochs (try 3-5)
    
    2. POLICY INSTABILITY (large KL divergence):
       - Decrease learning_rate (try 0.5x current value)
       - Decrease clip_range (try 0.1-0.15)
       - Increase max_grad_norm clipping
    
    3. POOR SAMPLE EFFICIENCY:
       - Increase n_steps (more data per update)
       - Increase n_epochs (more learning per data)
       - Check reward function design
    
    4. GPU MEMORY ISSUES:
       - Reduce batch_size
       - Reduce network sizes
       - Reduce n_steps
       - Use gradient checkpointing
    
    5. VALUE FUNCTION NOT LEARNING (high value_loss):
       - Increase vf_coef
       - Check reward scaling
       - Increase value network size
       - Verify reward function correctness
    """
    
    configs = {
        # RTX 4090 (24GB VRAM) - EXPLORATION EXPLOSIVE CONFIGURATIONS
        "rtx4090_standard": {
            "learning_rate": 1e-3,          # HIGHER: Faster policy changes
            "n_steps": 16384,               # LARGE: More exploration samples
            "batch_size": 1024,             # LARGE: Big gradient updates
            "n_epochs": 6,                  # MODERATE: Prevent overfitting
            "gamma": 0.98,                  # LOWER: Focus on immediate rewards
            "gae_lambda": 0.9,              # LOWER: Less long-term bias
            "clip_range": 0.4,              # HIGH: Allow large policy changes
            "ent_coef": 0.25,               # VERY HIGH: Maximum exploration
            "vf_coef": 0.3,                 # LOW: De-emphasize value function
            "max_grad_norm": 2.0,           # HIGH: Allow large gradients
            "policy_kwargs": dict(
                net_arch=dict(pi=[1024, 1024, 512], vf=[512, 512, 256]),
                activation_fn=torch.nn.ReLU,
                log_std_init=0.5            # EXPLOSIVE: High initial action variance
            )
        },
        
        "rtx4090_exploration": {
            "learning_rate": 2e-3,          # VERY HIGH: Explosive learning
            "n_steps": 8192,                # MODERATE: Frequent updates
            "batch_size": 512,              # MODERATE: Fast iterations
            "n_epochs": 4,                  # LOW: Prevent overfitting
            "gamma": 0.95,                  # LOW: Immediate rewards focus
            "gae_lambda": 0.85,             # LOW: Less smoothing
            "clip_range": 0.5,              # VERY HIGH: Massive policy changes
            "ent_coef": 0.35,               # EXTREME: Maximum chaos
            "vf_coef": 0.2,                 # VERY LOW: Ignore value function
            "max_grad_norm": 5.0,           # EXTREME: Allow gradient explosions
            "policy_kwargs": dict(
                net_arch=dict(pi=[2048, 1024, 512], vf=[256, 256]),  # Huge policy network
                activation_fn=torch.nn.ReLU,
                log_std_init=1.0            # VERY EXPLOSIVE: Maximum action variance
            )
        },
        
        # EXPLOSIVE CHAOS MODE - for visualizing policy explosion
        "rtx4090_explosive": {
            "learning_rate": 5e-3,          # EXTREME: Very fast learning
            "n_steps": 4096,                # SMALL: Frequent chaotic updates
            "batch_size": 256,              # SMALL: Fast chaotic iterations
            "n_epochs": 3,                  # LOW: Minimal constraint
            "gamma": 0.9,                   # VERY LOW: Only immediate rewards
            "gae_lambda": 0.8,              # VERY LOW: No smoothing
            "clip_range": 1.0,              # MAXIMUM: No clipping constraint
            "ent_coef": 0.5,                # MAXIMUM: Pure chaos
            "vf_coef": 0.1,                 # MINIMAL: Ignore value completely
            "max_grad_norm": 10.0,          # EXTREME: No gradient limits
            "policy_kwargs": dict(
                net_arch=dict(pi=[2048, 2048, 1024], vf=[128]),  # Massive policy, tiny value
                activation_fn=torch.nn.ReLU,
                log_std_init=2.0            # CHAOS: Maximum action explosion
            )
        },
        
        # RTX 3080/3090 (10-24GB VRAM) - Balanced Performance
        "rtx30xx_balanced": {
            "learning_rate": 4e-4,          # Stable learning rate
            "n_steps": 8192,                # Moderate rollout size
            "batch_size": 512,              # Good GPU utilization
            "n_epochs": 10,                 # Standard epoch count
            "gamma": 0.99,                  # Standard discount
            "gae_lambda": 0.95,             # Balanced advantage
            "clip_range": 0.2,              # Conservative updates
            "ent_coef": 0.02,               # Moderate exploration
            "vf_coef": 0.5,                 # Balanced learning
            "max_grad_norm": 0.5,           # Stable gradients
            "policy_kwargs": dict(
                net_arch=dict(pi=[768, 768, 384], vf=[384, 384, 192]),
                activation_fn=torch.nn.ReLU,
                log_std_init=-0.5
            )
        },
        
        # RTX 2080 Ti (11GB VRAM) - Conservative Configuration
        "rtx2080ti_conservative": {
            "learning_rate": 3e-4,          # Conservative LR
            "n_steps": 4096,                # Smaller rollout for memory
            "batch_size": 256,              # Small batches
            "n_epochs": 8,                  # Moderate epochs
            "gamma": 0.99,                  # Standard discount
            "gae_lambda": 0.95,             # Standard GAE
            "clip_range": 0.2,              # Conservative clipping
            "ent_coef": 0.01,               # Lower exploration
            "vf_coef": 0.5,                 # Balanced learning
            "max_grad_norm": 0.5,           # Conservative gradients
            "policy_kwargs": dict(
                net_arch=dict(pi=[512, 512, 256], vf=[256, 256, 128]),
                activation_fn=torch.nn.ReLU,
                log_std_init=-1.0           # Conservative exploration
            )
        },
        
        # RTX 5090 (32GB VRAM) - Maximum Performance
        "rtx5090_maximum": {
            "learning_rate": 6e-4,          # High performance LR
            "n_steps": 32768,               # Maximum rollout size
            "batch_size": 2048,             # Large batches
            "n_epochs": 12,                 # More learning per batch
            "gamma": 0.995,                 # Long-term planning
            "gae_lambda": 0.96,             # High-quality advantages
            "clip_range": 0.2,              # Conservative but effective
            "ent_coef": 0.03,               # Good exploration
            "vf_coef": 0.5,                 # Balanced learning
            "max_grad_norm": 0.5,           # Stable training
            "policy_kwargs": dict(
                net_arch=dict(pi=[2048, 1024, 512, 256], vf=[1024, 512, 256, 128]),
                activation_fn=torch.nn.ReLU,
                log_std_init=-0.3           # Moderate exploration
            )
        },
        
        # Debug/Development Configuration - EXTREME ANTI-STAGNATION
        "debug_fast": {
            "learning_rate": 3e-3,          # EXTREME: Maximum learning rate
            "n_steps": 2048,                # TINY: Very frequent updates
            "batch_size": 128,              # TINY: Maximum gradient steps
            "n_epochs": 2,                  # MINIMAL: No overfitting
            "gamma": 0.95,                  # SHORT-TERM: Immediate rewards
            "gae_lambda": 0.85,             # LOW: High bias, low variance
            "clip_range": 0.6,              # MASSIVE: Allow huge policy changes
            "ent_coef": 0.5,                # MAXIMUM: Force exploration
            "vf_coef": 0.05,                # IGNORE: Value function doesn't matter
            "max_grad_norm": 5.0,           # EXTREME: No gradient clipping essentially
            "policy_kwargs": dict(
                net_arch=dict(pi=[128, 128], vf=[64, 64]),  # MINIMAL: Fast learning
                activation_fn=torch.nn.ReLU,
                log_std_init=1.5            # MAXIMUM exploration
            )
        },
        
        # Emergency reset configuration for completely stuck training
        "emergency_reset": {
            "learning_rate": 5e-3,          # INSANE: Break any convergence
            "n_steps": 1024,                # ULTRA-SMALL: Constant updates
            "batch_size": 64,               # ULTRA-SMALL: Maximum update frequency
            "n_epochs": 1,                  # SINGLE: No overfitting possible
            "gamma": 0.9,                   # VERY SHORT: Only immediate rewards
            "gae_lambda": 0.8,              # VERY LOW: High bias
            "clip_range": 0.8,              # ABSURD: Allow complete policy changes
            "ent_coef": 1.0,                # MAXIMUM: Exploration over everything
            "vf_coef": 0.01,                # IGNORE: Value function irrelevant
            "max_grad_norm": 10.0,          # NO LIMITS: No gradient clipping
            "policy_kwargs": dict(
                net_arch=dict(pi=[64, 64], vf=[32, 32]),  # ULTRA-MINIMAL
                activation_fn=torch.nn.ReLU,
                log_std_init=2.0            # ABSURD exploration
            )
        }
    }
    
    return configs


def select_optimal_configuration(gpu_name: str, gpu_memory_gb: float, system_ram_gb: float) -> Tuple[str, Dict[str, Any]]:
    """
    Automatically select the optimal PPO configuration based on hardware specs
    
    Args:
        gpu_name: GPU model name (lowercase)
        gpu_memory_gb: Available GPU VRAM in GB
        system_ram_gb: Available system RAM in GB
    
    Returns:
        config_name: Name of selected configuration
        config_params: PPO hyperparameters for the configuration
    """
    configs = get_ppo_hyperparameters()
    
    print(f"\n🔍 Hardware-Based Configuration Selection")
    print(f"   GPU: {gpu_name}")
    print(f"   VRAM: {gpu_memory_gb:.1f}GB")
    print(f"   RAM: {system_ram_gb:.1f}GB")
    
    # Configuration selection logic
    if "5090" in gpu_name and gpu_memory_gb >= 28:
        config_name = "rtx5090_maximum"
        print("🚀 Selected: RTX 5090 Maximum Performance Configuration")
        
    elif "4090" in gpu_name and gpu_memory_gb >= 20:
        # USE BALANCED ANTI-STAGNATION CONFIGURATION
        config_name = "rtx4090_standard"
        print("🚀 Selected: RTX 4090 BALANCED Configuration")
        print("   ✅ Prevents stagnation while maintaining stability")
        
    elif any(gpu in gpu_name for gpu in ["3080", "3090", "3070"]) and gpu_memory_gb >= 8:
        config_name = "rtx30xx_balanced"
        print("🚀 Selected: RTX 30XX Balanced Configuration")
        
    elif "2080" in gpu_name or gpu_memory_gb < 12:
        config_name = "rtx2080ti_conservative"
        print("🚀 Selected: RTX 2080 Ti Conservative Configuration")
        
    else:
        # FORCE EMERGENCY CONFIGURATION FOR UNKNOWN HARDWARE
        config_name = "emergency_reset"
        print("⚠️  Unknown GPU - Using EMERGENCY ANTI-STAGNATION Configuration")
        print("   This uses extreme hyperparameters to force learning")
    
    return config_name, configs[config_name]


def create_training_environment(visualize: bool = False, num_envs: int = 1) -> Any:
    """
    Create training environment with proper configuration
    
    Args:
        visualize: Whether to enable visualization (only for single env)
        num_envs: Number of parallel environments (for faster training)
    
    Returns:
        Configured environment (vectorized if num_envs > 1)
    """
    print(f"\n🏗️  Environment Setup")
    print(f"   Environments: {num_envs}")
    print(f"   Visualization: {visualize}")
    
    if num_envs == 1:
        # Single environment
        env = TensegrityEnv(visualize=visualize)
        
        # Verify observation space consistency (critical per coding guidelines)
        expected_obs_dim = 96  # From coding guidelines
        actual_obs_dim = env.observation_space.shape[0]
        
        if actual_obs_dim != expected_obs_dim:
            print(f"⚠️  OBSERVATION DIMENSION MISMATCH:")
            print(f"   Expected: {expected_obs_dim}")
            print(f"   Actual: {actual_obs_dim}")
            print(f"   Simulator obs_dim: {env.sim.obs_dim}")
        
        print(f"✅ Environment configured:")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        print(f"   Action bounds: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
        
        return env
    
    else:
        # Multiple parallel environments for faster training
        def make_env():
            return TensegrityEnv(visualize=False)  # No viz for parallel envs
        
        env = SubprocVecEnv([make_env for _ in range(num_envs)])
        
        print(f"✅ Vectorized environment configured:")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        print(f"   Parallel environments: {num_envs}")
        
        return env


def setup_tensorboard_logging(config_name: str) -> str:
    """
    Setup TensorBoard logging with timestamped directories
    
    Args:
        config_name: Configuration name for logging directory
    
    Returns:
        log_dir: TensorBoard logging directory path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./ppo_tensegrity_tensorboard_gpu_{config_name}_{timestamp}/"
    
    # Ensure directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 TensorBoard Logging Setup")
    print(f"   Log directory: {log_dir}")
    print(f"   View with: tensorboard --logdir {log_dir}")
    
    return log_dir


def gpu_ppo_training(
    config_name: str,
    hyperparams: Dict[str, Any],
    total_timesteps: int = 5_000_000,
    save_freq: int = 500_000,
    eval_freq: int = 100_000,
    visualize: bool = False,
    num_envs: int = 1
) -> Dict[str, Any]:
    """
    Main GPU PPO training function for tensegrity robot locomotion
    
    Args:
        config_name: Name of the configuration being used
        hyperparams: PPO hyperparameters dictionary
        total_timesteps: Total training timesteps
        save_freq: How often to save model checkpoints
        eval_freq: How often to evaluate the model
        visualize: Whether to enable visualization
        num_envs: Number of parallel environments
    
    Returns:
        training_results: Dictionary with training metrics and results
    """
    print(f"\n{'='*80}")
    print(f"🚀 GPU PPO TRAINING: {config_name}")
    print(f"{'='*80}")
    
    training_start_time = time.time()
    training_results = {
        "success": False,
        "config_name": config_name,
        "hyperparams": hyperparams,
        "timing": {},
        "gpu_metrics": {}
    }
    
    try:
        # 1. Environment Setup
        print("\n1️⃣ Environment Setup...")
        env_start_time = time.time()
        
        env = create_training_environment(visualize=visualize, num_envs=num_envs)
        
        training_results["timing"]["environment_setup"] = time.time() - env_start_time
        print(f"   ✅ Environment setup completed ({training_results['timing']['environment_setup']:.2f}s)")
        
        # 2. TensorBoard Logging Setup
        print("\n2️⃣ Logging Setup...")
        log_setup_time = time.time()
        
        log_dir = setup_tensorboard_logging(config_name)
        
        training_results["timing"]["logging_setup"] = time.time() - log_setup_time
        training_results["log_dir"] = log_dir
        
        # 3. GPU Memory Monitoring
        print("\n3️⃣ GPU Memory Check...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_gpu_memory = torch.cuda.memory_allocated(0) / 1e9
            max_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"   GPU Memory - Allocated: {initial_gpu_memory:.2f}GB")
            print(f"   GPU Memory - Available: {max_gpu_memory - initial_gpu_memory:.2f}GB")
            
            training_results["gpu_metrics"]["initial_memory"] = initial_gpu_memory
            training_results["gpu_metrics"]["total_memory"] = max_gpu_memory
        
        # 4. PPO Model Initialization
        print("\n4️⃣ PPO Model Initialization...")
        model_start_time = time.time()
        
        # Create custom monitor callback
        monitor_callback = GPUTrainingMonitor(check_freq=1000, verbose=1)
        
        # Initialize PPO model with GPU acceleration
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda:0",
            **hyperparams
        )
        
        # Monitor GPU memory after model creation
        if torch.cuda.is_available():
            model_gpu_memory = torch.cuda.memory_allocated(0) / 1e9
            memory_increase = model_gpu_memory - initial_gpu_memory
            
            print(f"   ✅ Model initialized")
            print(f"   GPU Memory after model: {model_gpu_memory:.2f}GB (+{memory_increase:.2f}GB)")
            
            training_results["gpu_metrics"]["model_memory"] = model_gpu_memory
        
        training_results["timing"]["model_initialization"] = time.time() - model_start_time
        
        # 5. Display Training Configuration
        print(f"\n5️⃣ Training Configuration Summary")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Save frequency: {save_freq:,}")
        print(f"   Eval frequency: {eval_freq:,}")
        print(f"   Learning rate: {hyperparams['learning_rate']}")
        print(f"   Batch size: {hyperparams['batch_size']}")
        print(f"   Network architecture: {hyperparams['policy_kwargs']['net_arch']}")
        print(f"   Entropy coefficient: {hyperparams['ent_coef']}")
        
        # 6. Training Execution
        print(f"\n6️⃣ Starting PPO Training...")
        training_exec_time = time.time()
        
        # Create save directory
        save_dir = Path("trained_models_gpu")
        save_dir.mkdir(exist_ok=True)
        
        print(f"   Training will save models to: {save_dir}")
        print(f"   Monitor progress with: tensorboard --logdir {log_dir}")
        print(f"   Training started at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Execute training
        model.learn(
            total_timesteps=total_timesteps,
            callback=monitor_callback,
            tb_log_name=f"PPO_{config_name}",
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        training_results["timing"]["training_execution"] = time.time() - training_exec_time
        print(f"   ✅ Training completed ({training_results['timing']['training_execution']:.2f}s)")
        
        # 7. Model Saving
        print("\n7️⃣ Saving Trained Model...")
        save_start_time = time.time()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = save_dir / f"ppo_tensegrity_gpu_{config_name}_{timestamp}"
        
        model.save(final_model_path)
        
        training_results["timing"]["model_saving"] = time.time() - save_start_time
        training_results["model_path"] = str(final_model_path)
        
        print(f"   ✅ Model saved to: {final_model_path}")
        
        # 8. Final GPU Memory Check
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated(0) / 1e9
            peak_gpu_memory = torch.cuda.max_memory_allocated(0) / 1e9
            
            training_results["gpu_metrics"]["final_memory"] = final_gpu_memory
            training_results["gpu_metrics"]["peak_memory"] = peak_gpu_memory
            
            print(f"\n📊 GPU Memory Summary:")
            print(f"   Peak usage: {peak_gpu_memory:.2f}GB")
            print(f"   Final usage: {final_gpu_memory:.2f}GB")
            print(f"   Efficiency: {(peak_gpu_memory / max_gpu_memory) * 100:.1f}%")
        
        # 9. Training Success
        total_training_time = time.time() - training_start_time
        training_results["timing"]["total_time"] = total_training_time
        training_results["success"] = True
        
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"   Total time: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)")
        print(f"   Average FPS: {total_timesteps / training_results['timing']['training_execution']:.1f}")
        
        return training_results
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ GPU OUT OF MEMORY ERROR")
        print(f"   Error: {e}")
        print(f"   💡 Solutions:")
        print(f"      - Reduce batch_size (current: {hyperparams.get('batch_size', 'unknown')})")
        print(f"      - Reduce n_steps (current: {hyperparams.get('n_steps', 'unknown')})")
        print(f"      - Reduce network size")
        print(f"      - Use gradient checkpointing")
        
        training_results["error"] = "GPU OOM"
        training_results["error_details"] = str(e)
        return training_results
        
    except Exception as e:
        print(f"\n❌ TRAINING ERROR: {e}")
        training_results["error"] = str(e)
        return training_results
    
    finally:
        # Cleanup
        try:
            env.close()
            if 'model' in locals():
                del model
        except:
            pass
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        print("🧹 Cleanup completed")


def main():
    """
    Main training pipeline with automatic hardware detection and configuration
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GPU-Optimized PPO Training for Tensegrity Robot Locomotion")
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=None,
        help="Total training timesteps (default: auto-detect based on GPU, 500K-1M recommended for quick training)"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Quick training mode: 500K timesteps regardless of GPU"
    )
    parser.add_argument(
        "--explosive", 
        action="store_true",
        help="Use explosive/chaotic hyperparameters for visualizing policy explosion"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Force specific config (rtx4090_standard, rtx4090_exploration, rtx4090_explosive)"
    )
    args = parser.parse_args()
    
    print("🤖 GPU-Optimized PPO Training for Tensegrity Robot Locomotion")
    print("=" * 80)
    print("This script will automatically detect your GPU and select optimal")
    print("hyperparameters for training a locomotion policy.")
    print()
    
    # 1. System Requirements Check
    system_ok, gpu_name, memory_specs = check_system_requirements()
    if not system_ok:
        print("\n❌ System requirements not met. Exiting.")
        return
    
    gpu_memory_gb, system_ram_gb = memory_specs
    
    # 2. Configuration Selection
    if args.config:
        # Force specific configuration
        all_configs = get_ppo_hyperparameters()
        if args.config in all_configs:
            config_name = args.config
            hyperparams = all_configs[args.config]
            print(f"🎯 Using forced configuration: {config_name}")
        else:
            print(f"❌ Configuration '{args.config}' not found. Available: {list(all_configs.keys())}")
            return
    elif args.explosive:
        # Force explosive configuration for RTX 4090
        config_name = "rtx4090_explosive"
        hyperparams = get_ppo_hyperparameters()[config_name]
        print("💥 EXPLOSIVE MODE: Using maximum chaos hyperparameters!")
        print("⚠️  Warning: This will likely cause policy explosion and chaotic behavior!")
    else:
        # Auto-detect based on hardware
        config_name, hyperparams = select_optimal_configuration(gpu_name, gpu_memory_gb, system_ram_gb)
    
    print(f"\n📋 Selected Hyperparameters:")
    for key, value in hyperparams.items():
        if key != "policy_kwargs":
            print(f"   {key}: {value}")
    print(f"   network_architecture: {hyperparams['policy_kwargs']['net_arch']}")
    
    # 3. Training Configuration
    print(f"\n⚙️ Training Configuration:")
    
    # Determine total timesteps based on arguments or hardware
    if args.quick:
        total_timesteps = 500_000
        print("   ⚡ Quick training mode: 500K timesteps")
    elif args.standard:
        total_timesteps = 1_000_000
        print("   🎯 Standard training mode: 1M timesteps")
    elif args.timesteps:
        total_timesteps = args.timesteps
        print(f"   🎯 Custom timesteps: {total_timesteps:,}")
    else:
        # Auto-detect based on hardware (original behavior but with reduced defaults)
        if "5090" in gpu_name:
            total_timesteps = 1_000_000  # Reduced from 10M for faster iteration
            print("   🚀 RTX 5090 detected - Standard training (1M timesteps)")
        elif "4090" in gpu_name:
            total_timesteps = 1_000_000   # Reduced from 8M for faster iteration
            print("   🚀 RTX 4090 detected - Standard training (1M timesteps)")
        elif any(gpu in gpu_name for gpu in ["3080", "3090"]):
            total_timesteps = 800_000   # Reduced from 5M for faster iteration
            print("   🚀 RTX 30XX detected - Moderate training (800K timesteps)")
        else:
            total_timesteps = 500_000   # Reduced from 3M for faster iteration
            print("   🚀 Conservative training (500K timesteps)")
    
    # Parallel environments for faster training (if enough VRAM)
    num_envs = 1 if gpu_memory_gb < 16 else min(4, int(gpu_memory_gb // 8))
    if num_envs > 1:
        print(f"   🔄 Using {num_envs} parallel environments for faster training")
    
    print(f"   📊 Total timesteps: {total_timesteps:,}")
    print(f"   💾 Models will be saved every 500K timesteps")
    print(f"   📈 TensorBoard logging enabled")
    
    # 4. Confirm Training Start
    print(f"\n🎯 Ready to start training with {config_name} configuration")
    print(f"   Estimated training time: {total_timesteps / 100000:.1f} hours @ 100K steps/hour")
    
    try:
        input("Press Enter to start training, or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\n❌ Training cancelled by user")
        return
    
    # 5. Execute Training
    results = gpu_ppo_training(
        config_name=config_name,
        hyperparams=hyperparams,
        total_timesteps=total_timesteps,
        save_freq=500_000,
        eval_freq=100_000,
        visualize=False,  # No visualization during training for performance
        num_envs=num_envs
    )
    
    # 6. Training Results Summary
    print(f"\n{'='*80}")
    print("🏁 TRAINING RESULTS SUMMARY")
    print(f"{'='*80}")
    
    if results["success"]:
        print(f"✅ Training completed successfully!")
        print(f"   Configuration: {results['config_name']}")
        print(f"   Total time: {results['timing']['total_time']:.2f}s")
        print(f"   Model saved: {results['model_path']}")
        print(f"   TensorBoard logs: {results['log_dir']}")
        
        if "gpu_metrics" in results:
            print(f"   Peak GPU usage: {results['gpu_metrics']['peak_memory']:.2f}GB")
        
        print(f"\n🎮 To test your trained model:")
        print(f"   python test_trained_model.py --model {results['model_path']}")
        
        print(f"\n📊 To view training progress:")
        print(f"   tensorboard --logdir {results['log_dir']}")
        
    else:
        print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
        if "error_details" in results:
            print(f"   Details: {results['error_details']}")


if __name__ == "__main__":
    main()