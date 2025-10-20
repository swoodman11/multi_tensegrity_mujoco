# Test Directional Control - Updates Summary

## Changes Made to `test_trained_model_SAC.py`

### 1. **Added Angle Display** ✅
Now displays the angle relative to world +X axis for each direction test:

```python
# Calculate angle relative to world X-axis
angle_rad = np.arctan2(direction_vector[1], direction_vector[0])
angle_deg = np.degrees(angle_rad)
# Normalize to [0, 360)
if angle_deg < 0:
    angle_deg += 360

print(f"  Angle from world +X axis: {angle_deg:.1f}°")
```

**Example Output:**
```
Testing direction: Northeast (45°)
  Goal direction vector: [0.707, 0.707]
  Angle from world +X axis: 45.0°
```

### 2. **Fixed Visualization Issues** ✅
Creates a **fresh environment for each direction test** to ensure proper rendering:

**Before (Broken):**
```python
def test_directional_control(model, env, ...):
    # Reused same environment for all tests
    for direction in directions:
        obs, info = env.reset()  # Just reset, don't recreate
```

**After (Fixed):**
```python
def test_directional_control(model, ...):  # No env parameter
    # Create fresh environment for each direction
    for direction in directions:
        env = TensegrityEnv(visualize=visualize)
        obs, info = env.reset()
        
        # ... run test ...
        
        # Clean up after each test
        env.close()
        del env
```

**Why This Fixes The Issue:**
- MuJoCo/OpenGL state can become stale when reusing environments
- Creating fresh environments ensures proper physics and rendering initialization
- Properly closes resources between tests

### 3. **Enhanced Summary Output** ✅
Added detailed results table showing angle, distance, and efficiency per direction:

```
Detailed Results by Direction:
----------------------------------------------------------------------
North (0°)           | Angle:    0.0° | Distance:  0.551m | Efficiency:  56.0%
Northeast (45°)      | Angle:   45.0° | Distance:  1.229m | Efficiency:  50.0%
East (90°)           | Angle:   90.0° | Distance:  0.225m | Efficiency:  12.9%
...
```

### 4. **Updated Function Signature**
Removed `env` parameter since we create fresh environments internally:

**Before:**
```python
test_directional_control(model=model, env=env, steps_per_direction=500, visualize=True)
```

**After:**
```python
test_directional_control(model=model, steps_per_direction=500, visualize=True)
```

## Usage

### Test with Visualization (Watch Robot Move)
```bash
conda activate tensegrity_gnn
python test_trained_model_SAC.py --model models/gpu/sac_gpu_pretraining_rtx2080ti_32gb_efficient_20251019_011320 --test-directions
```

### Test without Visualization (Faster)
```bash
python test_trained_model_SAC.py --model <model_path> --test-directions --no-vis
```

## Expected Behavior

### With Visualization (`--test-directions` only)
- Opens MuJoCo viewer window for **each direction test**
- Window closes after each test (500 steps)
- New window opens for next direction
- Robot should be visible and moving in each test
- Camera follows robot as it moves

### Without Visualization (`--test-directions --no-vis`)
- No rendering windows
- Tests run faster
- Still prints all metrics

## Output Format

```
======================================================================
DIRECTIONAL CONTROL TEST
======================================================================

Testing direction: North (0°)
  Goal direction vector: [1.000, 0.000]
  Angle from world +X axis: 0.0°
10.0 0.2 2.0
  Total reward: -697.30
  Directional distance: 0.551m
  Perpendicular deviation: 0.432m
  Distance per step: 0.0011m
  Directional efficiency: 56.03%

[... 7 more directions ...]

======================================================================
SUMMARY
======================================================================
Average reward across all directions: -1330.21
Average directional distance: 0.468m
Average efficiency: -2145.48%

Detailed Results by Direction:
----------------------------------------------------------------------
North (0°)           | Angle:    0.0° | Distance:  0.551m | Efficiency:  56.0%
Northeast (45°)      | Angle:   45.0° | Distance:  1.229m | Efficiency:  50.0%
[... etc ...]
======================================================================
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'stable_baselines3'"
**Solution:** Activate conda environment first
```bash
conda activate tensegrity_gnn
```

### Issue: Robot not visible in visualization, camera just jerks around
**Status:** ✅ FIXED - Now creates fresh environment for each direction test

### Issue: Want to test fewer directions (for speed)
**Solution:** Modify the test to only include specific directions:
```python
# In test_trained_model_SAC.py, before running:
custom_directions = [
    ("North (0°)", np.array([1.0, 0.0])),
    ("East (90°)", np.array([0.0, 1.0])),
]
test_results = test_directional_control(
    model=model,
    directions_to_test=custom_directions,  # Custom list
    steps_per_direction=500,
    visualize=True
)
```

## Technical Details

### Angle Calculation
- Uses `np.arctan2(y, x)` to compute angle from direction vector
- Normalizes to [0°, 360°) range for consistency
- 0° = +X axis (East in standard coordinates)
- 90° = +Y axis (North)
- 180° = -X axis (West)
- 270° = -Y axis (South)

### Environment Lifecycle Per Test
1. Create `TensegrityEnv(visualize=True/False)`
2. Call `env.reset()` to initialize
3. Override `env.sim.goal_direction` with test direction
4. Run 500 steps with model predictions
5. Call `env.close()` to clean up
6. Delete environment object
7. Repeat for next direction

This ensures each test starts with a clean slate and proper rendering state.

## Files Modified
- ✅ `test_trained_model_SAC.py` - Updated `test_directional_control()` function

## Verification
Run the test without visualization to quickly verify functionality:
```bash
conda activate tensegrity_gnn
python test_trained_model_SAC.py --model models/gpu/sac_gpu_pretraining_rtx2080ti_32gb_efficient_20251019_011320 --test-directions --no-vis
```

Should complete 8 direction tests in ~2-3 minutes and show the summary table.
