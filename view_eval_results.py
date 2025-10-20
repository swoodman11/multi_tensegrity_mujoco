"""
View and analyze evaluation results from EvalCallback
This script loads and displays the data stored in evaluations.npz files
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_evaluation_data(eval_path):
    """
    Load evaluation data from evaluations.npz file
    
    Args:
        eval_path: Path to the evaluations.npz file or directory containing it
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    eval_path = Path(eval_path)
    
    # If directory provided, look for evaluations.npz inside
    if eval_path.is_dir():
        eval_file = eval_path / "evaluations.npz"
    else:
        eval_file = eval_path
    
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
    
    print(f"📂 Loading evaluation data from: {eval_file}")
    data = np.load(eval_file)
    
    return data


def print_evaluation_summary(data):
    """Print comprehensive summary of evaluation results"""
    
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("="*70)
    
    # Available keys in the npz file
    print(f"\n📋 Available data fields: {list(data.keys())}")
    
    # Timesteps when evaluations occurred
    if 'timesteps' in data:
        timesteps = data['timesteps']
        print(f"\n⏱️  Evaluation Timesteps:")
        print(f"   Total evaluations: {len(timesteps)}")
        print(f"   First evaluation: {timesteps[0]:,} steps")
        print(f"   Last evaluation: {timesteps[-1]:,} steps")
        print(f"   Evaluation frequency: ~{timesteps[1] - timesteps[0]:,} steps")
    
    # Episode rewards (mean reward per evaluation)
    if 'results' in data:
        results = data['results']
        print(f"\n🎯 Episode Rewards (Mean per evaluation):")
        print(f"   Shape: {results.shape} (evaluations × episodes)")
        print(f"   Number of episodes per eval: {results.shape[1]}")
        
        # Calculate statistics across all evaluations
        mean_rewards = np.mean(results, axis=1)
        std_rewards = np.std(results, axis=1)
        
        print(f"\n📈 Reward Statistics:")
        print(f"   Overall best mean reward: {np.max(mean_rewards):.2f}")
        print(f"   Overall worst mean reward: {np.min(mean_rewards):.2f}")
        print(f"   Final mean reward: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
        print(f"   Average across all evals: {np.mean(mean_rewards):.2f}")
        
        # Find best evaluation point
        best_idx = np.argmax(mean_rewards)
        print(f"\n🏆 Best Performance:")
        print(f"   Achieved at timestep: {timesteps[best_idx]:,}")
        print(f"   Mean reward: {mean_rewards[best_idx]:.2f} ± {std_rewards[best_idx]:.2f}")
        print(f"   Individual episode rewards: {results[best_idx]}")
    
    # Episode lengths
    if 'ep_lengths' in data:
        ep_lengths = data['ep_lengths']
        print(f"\n📏 Episode Lengths:")
        print(f"   Shape: {ep_lengths.shape}")
        mean_lengths = np.mean(ep_lengths, axis=1)
        print(f"   Average episode length: {np.mean(mean_lengths):.1f} steps")
        print(f"   Final episode length: {mean_lengths[-1]:.1f} steps")
    
    print("\n" + "="*70)


def plot_evaluation_results(data, save_path=None):
    """
    Create visualization plots of evaluation results
    
    Args:
        data: Loaded npz data
        save_path: Optional path to save the plot
    """
    if 'timesteps' not in data or 'results' not in data:
        print("⚠️  Required data fields not found for plotting")
        return
    
    timesteps = data['timesteps']
    results = data['results']
    
    # Calculate mean and std for each evaluation
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Mean reward over time with confidence interval
    ax1 = axes[0]
    ax1.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
    ax1.fill_between(timesteps, 
                      mean_rewards - std_rewards, 
                      mean_rewards + std_rewards, 
                      alpha=0.3, color='blue', label='±1 Std Dev')
    ax1.axhline(y=np.max(mean_rewards), color='r', linestyle='--', 
                linewidth=1, label=f'Best: {np.max(mean_rewards):.2f}')
    ax1.set_xlabel('Training Timesteps', fontsize=12)
    ax1.set_ylabel('Mean Episode Reward', fontsize=12)
    ax1.set_title('Training Progress: Evaluation Rewards', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot 2: Episode lengths over time
    if 'ep_lengths' in data:
        ax2 = axes[1]
        ep_lengths = data['ep_lengths']
        mean_lengths = np.mean(ep_lengths, axis=1)
        std_lengths = np.std(ep_lengths, axis=1)
        
        ax2.plot(timesteps, mean_lengths, 'g-', linewidth=2, label='Mean Episode Length')
        ax2.fill_between(timesteps, 
                          mean_lengths - std_lengths, 
                          mean_lengths + std_lengths, 
                          alpha=0.3, color='green', label='±1 Std Dev')
        ax2.set_xlabel('Training Timesteps', fontsize=12)
        ax2.set_ylabel('Episode Length (steps)', fontsize=12)
        ax2.set_title('Training Progress: Episode Lengths', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def export_to_csv(data, output_path):
    """
    Export evaluation data to CSV for further analysis
    
    Args:
        data: Loaded npz data
        output_path: Path to save CSV file
    """
    if 'timesteps' not in data or 'results' not in data:
        print("⚠️  Required data fields not found for CSV export")
        return
    
    timesteps = data['timesteps']
    results = data['results']
    
    # Prepare data for CSV
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    min_rewards = np.min(results, axis=1)
    max_rewards = np.max(results, axis=1)
    
    # Create header
    header = "timestep,mean_reward,std_reward,min_reward,max_reward"
    
    # Add episode lengths if available
    if 'ep_lengths' in data:
        ep_lengths = data['ep_lengths']
        mean_lengths = np.mean(ep_lengths, axis=1)
        header += ",mean_episode_length"
        csv_data = np.column_stack([timesteps, mean_rewards, std_rewards, 
                                     min_rewards, max_rewards, mean_lengths])
    else:
        csv_data = np.column_stack([timesteps, mean_rewards, std_rewards, 
                                     min_rewards, max_rewards])
    
    # Save to CSV
    np.savetxt(output_path, csv_data, delimiter=',', header=header, 
               comments='', fmt='%.6f')
    print(f"💾 CSV exported to: {output_path}")


def main():
    """Main function with command-line interface"""
    
    parser = argparse.ArgumentParser(
        description='View and analyze evaluation results from EvalCallback',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View evaluations from a specific config
  python view_eval_results.py --path ./logs/evals_rtx2080ti_32gb_efficient/
  
  # View and save plot
  python view_eval_results.py --path ./logs/evals_rtx4090_large/ --plot results.png
  
  # Export to CSV
  python view_eval_results.py --path ./logs/evals_rtx4090_large/ --csv results.csv
  
  # Do everything
  python view_eval_results.py --path ./logs/evals_rtx4090_large/ --plot plot.png --csv data.csv
        """
    )
    
    parser.add_argument(
        '--path', '-p',
        type=str,
        default='./logs/evals_rtx2080ti_32gb_efficient/',
        help='Path to evaluations.npz file or directory containing it'
    )
    
    parser.add_argument(
        '--plot',
        type=str,
        default=None,
        help='Save plot to specified path (e.g., results.png)'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Export data to CSV file (e.g., results.csv)'
    )
    
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List all available evaluation log directories'
    )
    
    args = parser.parse_args()
    
    # List available configs if requested
    if args.list_configs:
        print("\n📁 Available evaluation log directories:")
        print("="*70)
        logs_dir = Path('./logs')
        if logs_dir.exists():
            eval_dirs = sorted(logs_dir.glob('evals_*'))
            if eval_dirs:
                for i, eval_dir in enumerate(eval_dirs, 1):
                    eval_file = eval_dir / 'evaluations.npz'
                    status = "✅" if eval_file.exists() else "❌"
                    print(f"{i:2d}. {status} {eval_dir.name}")
            else:
                print("   No evaluation directories found")
        else:
            print("   ./logs directory not found")
        print("="*70)
        return
    
    try:
        # Load data
        data = load_evaluation_data(args.path)
        
        # Print summary
        print_evaluation_summary(data)
        
        # Generate plot if requested
        if args.plot:
            plot_evaluation_results(data, args.plot)
        
        # Export to CSV if requested
        if args.csv:
            export_to_csv(data, args.csv)
        
        # If no output specified, show interactive plot
        if not args.plot and not args.csv:
            print("\n📊 Displaying interactive plot...")
            plot_evaluation_results(data, save_path=None)
        
        print("\n✅ Analysis complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tip: Use --list-configs to see available evaluation directories")
        print("   Or check that your training has run long enough to generate evaluations")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
