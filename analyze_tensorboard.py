"""
Analyze TensorBoard logs and export key metrics for review
"""

from tensorboard.backend.event_processing import event_accumulator
import os
import glob

def analyze_tensorboard_run(log_dir):
    """
    Extract and print key metrics from TensorBoard logs
    
    Args:
        log_dir: Path to tensorboard log directory (e.g., './sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient/SAC_31')
    """
    
    # Find the most recent run
    if not os.path.exists(log_dir):
        # Try to find the latest SAC run
        base_dir = os.path.dirname(log_dir) if os.path.dirname(log_dir) else '.'
        pattern = os.path.join(base_dir, 'SAC_*')
        runs = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        if runs:
            log_dir = runs[-1]
            print(f"Using most recent run: {log_dir}")
        else:
            print(f"ERROR: No runs found matching pattern: {pattern}")
            return
    
    # Load the event file
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()['scalars']
    
    print("="*80)
    print("TENSORBOARD METRICS ANALYSIS")
    print("="*80)
    print(f"\nLog directory: {log_dir}")
    print(f"Available metrics: {len(tags)}")
    
    # Key metrics to extract
    key_metrics = [
        'rollout/ep_rew_mean',
        'rollout/ep_len_mean',
        'train/actor_loss',
        'train/critic_loss',
        'train/ent_coef',
        'train/learning_rate',
        'time/fps',
    ]
    
    print("\n" + "="*80)
    print("KEY METRICS SUMMARY")
    print("="*80)
    
    for metric in key_metrics:
        if metric in tags:
            events = ea.Scalars(metric)
            if events:
                print(f"\n{metric}:")
                print(f"  {'Step':<10} {'Value':<15} {'Wall Time':<20}")
                print(f"  {'-'*10} {'-'*15} {'-'*20}")
                
                # Show first, middle, and last values
                indices = [0]
                if len(events) > 2:
                    indices.append(len(events) // 2)
                if len(events) > 1:
                    indices.append(-1)
                
                for idx in indices:
                    event = events[idx]
                    print(f"  {event.step:<10} {event.value:<15.4f} {event.wall_time:.2f}")
                
                # Statistics
                values = [e.value for e in events]
                print(f"\n  Min: {min(values):.4f}")
                print(f"  Max: {max(values):.4f}")
                print(f"  Final: {values[-1]:.4f}")
                print(f"  Trend: {'↑ Improving' if values[-1] > values[0] else '↓ Degrading' if values[-1] < values[0] else '→ Stable'}")
        else:
            print(f"\n{metric}: NOT FOUND")
    
    # Print all available metrics
    print("\n" + "="*80)
    print("ALL AVAILABLE METRICS")
    print("="*80)
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        if events:
            print(f"{tag:<50} {len(events)} samples, final value: {events[-1].value:.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Export to CSV (optional)
    try:
        import pandas as pd
        
        export_dir = "tensorboard_exports"
        os.makedirs(export_dir, exist_ok=True)
        
        for metric in key_metrics:
            if metric in tags:
                events = ea.Scalars(metric)
                df = pd.DataFrame([{
                    'step': e.step,
                    'value': e.value,
                    'wall_time': e.wall_time
                } for e in events])
                
                filename = os.path.join(export_dir, f"{metric.replace('/', '_')}.csv")
                df.to_csv(filename, index=False)
                print(f"Exported: {filename}")
        
        print(f"\n✅ CSV files saved to: {export_dir}/")
    except ImportError:
        print("\n⚠️  Install pandas to export CSVs: pip install pandas")


if __name__ == "__main__":
    import sys
    
    # Default to the SAC tensorboard directory
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient"
    
    analyze_tensorboard_run(log_dir)
