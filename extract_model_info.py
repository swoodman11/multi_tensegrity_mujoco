"""
Extract hyperparameters and network architecture from saved SAC models.
"""

import sys
from pathlib import Path
from stable_baselines3 import SAC

def extract_model_info(model_path):
    """Extract and print model hyperparameters and architecture."""
    print(f"\n{'='*80}")
    print(f"MODEL: {Path(model_path).name}")
    print(f"{'='*80}\n")
    
    try:
        # Load model
        model = SAC.load(model_path, device='cpu')
        
        # ===== SAC HYPERPARAMETERS =====
        print("SAC HYPERPARAMETERS:")
        print("-" * 40)
        print(f"Learning Rate:        {model.learning_rate}")
        print(f"Buffer Size:          {model.buffer_size:,}")
        print(f"Batch Size:           {model.batch_size}")
        print(f"Gamma (discount):     {model.gamma}")
        print(f"Tau (target update):  {model.tau}")
        
        # Entropy coefficient
        if hasattr(model, 'ent_coef'):
            if isinstance(model.ent_coef, float):
                print(f"Entropy Coefficient:  {model.ent_coef}")
            else:
                print(f"Entropy Coefficient:  auto (current: {model.ent_coef_tensor.item():.4f})")
        
        if hasattr(model, 'target_entropy'):
            print(f"Target Entropy:       {model.target_entropy}")
        
        print(f"Training Timesteps:   {model.num_timesteps:,}")
        
        # ===== NETWORK ARCHITECTURE =====
        print(f"\nNETWORK ARCHITECTURE:")
        print("-" * 40)
        
        # Actor network
        print("\nActor Network:")
        print(model.actor)
        
        # Critic network
        print("\nCritic Network:")
        print(model.critic)
        
        # Feature extractor details
        if hasattr(model.policy, 'features_extractor'):
            print("\nFeatures Extractor:")
            print(model.policy.features_extractor)
        
        # Action and observation spaces
        print(f"\nAction Space:         {model.action_space}")
        print(f"Observation Space:    {model.observation_space}")
        
        # ===== ADDITIONAL INFO =====
        print(f"\nADDITIONAL INFO:")
        print("-" * 40)
        
        # Check for any custom attributes
        custom_attrs = [attr for attr in dir(model) if not attr.startswith('_') 
                       and attr not in ['actor', 'critic', 'policy', 'action_space', 
                                       'observation_space', 'learning_rate', 'gamma', 
                                       'tau', 'buffer_size', 'batch_size']]
        
        if custom_attrs:
            print("Custom attributes found:")
            for attr in custom_attrs[:10]:  # Limit to first 10
                try:
                    val = getattr(model, attr)
                    if not callable(val):
                        print(f"  {attr}: {val}")
                except:
                    pass
        
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"❌ ERROR loading model: {e}\n")

if __name__ == "__main__":
    # Default models to check
    models_to_check = [
        "models/gpu/sac_gpu_pretraining_rtx2080ti_32gb_efficient_20251019_223958.zip",  # SAC_34
        "models/gpu/sac_gpu_pretraining_rtx2080ti_32gb_efficient_20251020_060135.zip",  # Latest
    ]
    
    # Allow command line argument
    if len(sys.argv) > 1:
        models_to_check = [sys.argv[1]]
    
    for model_path in models_to_check:
        if Path(model_path).exists():
            extract_model_info(model_path)
        else:
            print(f"❌ Model not found: {model_path}\n")
