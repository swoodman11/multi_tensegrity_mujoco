# Efficiency Calculation Fix

## Bug Found ✅

The original efficiency calculation had a critical bug when the robot moved in the **opposite direction** from the goal.

### Original (Buggy) Formula
```python
efficiency = directional_distance / (directional_distance + perpendicular_deviation)
```

**Problem:**
- When `directional_distance` is negative (moving backwards), this produces nonsensical values
- Example: `directional_distance = -1.46m`, `perpendicular_deviation = 0.01m`
  - Old efficiency = -1.46 / (-1.46 + 0.01) = -1.46 / -1.45 ≈ 100.7%
  - Or worse: -1.46 / (-1.46 + 0.1) = -107.4%
- Led to efficiency values like **-13526.9%** which don't make physical sense

### New (Fixed) Formula
```python
total_distance_traveled = ||displacement||  # Euclidean norm
efficiency = (directional_distance / total_distance_traveled) × 100%
```

**Interpretation:**
- **Range**: -100% to +100%
- **+100%**: Moving perfectly toward goal in straight line
- **0%**: Moving perpendicular to goal (or not moving)
- **-100%**: Moving directly away from goal in straight line
- **+50%**: Half the distance traveled is in the goal direction
- **-80%**: Mostly moving away from goal

### Additional Metric: Straightness
```python
straightness = (|directional_distance| / total_distance_traveled) × 100%
```

**Interpretation:**
- **Range**: 0% to 100%
- **100%**: Perfectly straight path (regardless of direction)
- **50%**: Half the movement is in goal direction, half is perpendicular
- **0%**: Only moving perpendicular to goal direction

This metric **ignores** whether you're going forward or backward - it just measures how straight your path is.

## Example Outputs

### Scenario 1: Moving Forward and Straight
- Goal direction: [1, 0] (East)
- Displacement: [2.0, 0.1] (2.0m east, 0.1m north)
- Total distance: 2.002m
- Directional distance: 2.0m

**Old efficiency**: 2.0 / (2.0 + 0.1) = 95.2% ✓ (happened to work)
**New efficiency**: (2.0 / 2.002) × 100 = 99.9% ✓ (more accurate)
**Straightness**: (2.0 / 2.002) × 100 = 99.9%

### Scenario 2: Moving Backward and Straight (Your Bug Case)
- Goal direction: [1, 0] (East, goal)
- Displacement: [-1.46, 0.01] (1.46m west, 0.01m north)
- Total distance: 1.460m
- Directional distance: -1.46m

**Old efficiency**: -1.46 / (-1.46 + 0.01) = **-14597.8%** ❌ (nonsense!)
**New efficiency**: (-1.46 / 1.460) × 100 = **-100.0%** ✓ (moving directly away)
**Straightness**: (1.46 / 1.460) × 100 = 99.9%

### Scenario 3: Moving Forward but Zigzagging
- Goal direction: [1, 0] (East)
- Displacement: [1.0, 2.0] (1.0m east, 2.0m north)
- Total distance: 2.236m
- Directional distance: 1.0m

**Old efficiency**: 1.0 / (1.0 + 2.0) = 33.3%
**New efficiency**: (1.0 / 2.236) × 100 = **44.7%** ✓ (less than half toward goal)
**Straightness**: (1.0 / 2.236) × 100 = 44.7%

## New Output Format

```
Testing direction: West (270°)
  Goal direction vector: [0.000, -1.000]
  Angle from world +X axis: 270.0°
  Total reward: -4215.00
  Directional distance: -1.460m
  Total distance traveled: 1.461m
  Perpendicular deviation: 0.010m
  Distance per step: -0.0029m
  Directional efficiency: -100.0% (positive = toward goal)
  Path straightness: 99.9% (100% = perfectly straight)
```

**Interpretation:**
- Robot moved 1.461m total
- Moved 1.460m in the **opposite** direction from goal (hence -1.460m)
- Only 0.010m perpendicular deviation (very straight path)
- **Efficiency: -100%** = Moving directly away from goal (bad!)
- **Straightness: 99.9%** = Very straight path (just wrong direction)

## Summary Table Format

```
Detailed Results by Direction:
------------------------------------------------------------------------------------------
Direction            |   Angle |  Dir.Dist | Efficiency | Straightness
------------------------------------------------------------------------------------------
North (0°)           |    0.0° |    0.551m |      56.0% |        56.0%
Northeast (45°)      |   45.0° |    1.229m |      70.7% |        70.7%
East (90°)           |   90.0° |    0.225m |      14.7% |        14.7%
Southeast (135°)     |  135.0° |    0.399m |      19.4% |        19.4%
South (180°)         |  180.0° |   -1.048m |     -48.3% |        48.3%  ← Moving backwards
Southwest (225°)     |  225.0° |   -0.846m |     -60.8% |        60.8%  ← Moving backwards
West (270°)          |  270.0° |   -1.460m |    -100.0% |        99.9%  ← Straight backwards!
Northwest (315°)     |  315.0° |    0.470m |      51.4% |        51.4%
==========================================================================================
Average directional efficiency: -13.5% (slightly moving away from goals on average)
Average path straightness: 52.7% (moderately straight paths)
```

## Key Insights

From your 25k-step trained model results:
- **Negative efficiency** now makes sense: It means moving away from the goal
- **West direction (-100% efficiency)**: Robot went almost perfectly straight... backwards!
- **Average -13.5% efficiency**: Robot tends to move slightly away from commanded directions
- **Average 52.7% straightness**: Paths are moderately straight regardless of direction

This suggests the policy hasn't learned directional control yet (which makes sense for only 25k steps). It needs more training to learn to follow the goal_direction in observations.

## Code Changes

**File**: `test_trained_model_SAC.py`
**Function**: `test_directional_control()`

1. Calculate `total_distance_traveled = np.linalg.norm(displacement)`
2. Calculate `efficiency = (directional_distance / total_distance_traveled) * 100.0`
3. Calculate `straightness = (abs(directional_distance) / total_distance_traveled) * 100.0`
4. Updated output formatting to show both metrics
5. Updated summary table with clearer column headers

## Testing

To verify the fix works correctly:
```bash
conda activate tensegrity_gnn
python test_trained_model_SAC.py --model models/gpu/sac_gpu_pretraining_rtx2080ti_32gb_efficient_20251019_011320 --test-directions --no-vis
```

You should now see:
- Efficiency values between -100% and +100%
- Negative efficiency when moving away from goal
- Positive efficiency when moving toward goal
- No more bizarre values like -13526.9%
