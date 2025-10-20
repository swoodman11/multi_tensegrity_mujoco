# Domain Randomization - Friction Disabled

## Change Made

**Date**: October 20, 2025

**Location**: `mujoco_physics_engine/tensegrity_mjc_simulation.py` - `reset()` method

**What Changed**:
- ✅ Friction randomization (±20%) is now **DISABLED**
- ✅ Goal direction randomization (0-360°) remains **ACTIVE**

## Rationale

To isolate the effects of reward weight rebalancing on gliding behavior:

1. **Friction variations** could mask whether weight changes are working
2. **Consistent friction** makes it easier to evaluate if robot is still gliding
3. **Cleaner testing** - one variable at a time

## Current Domain Randomization

### Active:
- ✅ **Goal direction**: Random angle 0-360° each reset
- ✅ **Action noise**: σ=0.05 (if enabled in training script)

### Disabled:
- ❌ **Friction**: Fixed at XML default values
- ❌ **Mass**: Not randomized
- ❌ **Other physics**: Not randomized

## Re-enabling Friction Later

Once reward rebalancing is validated and gliding is fixed:

```python
# In tensegrity_mjc_simulation.py, reset() method, line ~176
# Uncomment these lines:

# Randomize friction (±20%)
friction_multiplier = np.random.uniform(0.8, 1.2)
for geom_id in range(self.mjc_model.ngeom):
    # Store original friction on first reset
    if not hasattr(self, '_original_friction'):
        self._original_friction = self.mjc_model.geom_friction.copy()
    
    # Apply randomization
    self.mjc_model.geom_friction[geom_id, 0] = \
        self._original_friction[geom_id, 0] * friction_multiplier
```

## Testing Protocol

### Phase 1: Weight Rebalancing (NOW)
- Friction: **OFF**
- Weight changes: **ACTIVE**
- Goal: Fix gliding with reward adjustments

```powershell
python gpu_pretraining_SAC.py --total-timesteps 25000
```

### Phase 2: Validation (After gliding fixed)
- Friction: **OFF** 
- Train full 500k to verify stable locomotion

### Phase 3: Robustness (Final)
- Friction: **ON**
- Train with randomization for sim-to-real transfer

## Expected Impact

### Positive:
- ✅ Easier to spot if gliding persists (no friction excuses)
- ✅ Faster debugging iterations
- ✅ More consistent training initially

### Trade-offs:
- ⚠️ Policy may overfit to specific friction value
- ⚠️ Will need retraining with randomization for robustness
- ⚠️ Less sim-to-real transfer capability initially

## Monitoring

### TensorBoard - No Change Expected
Disabling friction randomization should NOT affect:
- Learning curves (may be slightly smoother)
- Convergence speed (may be slightly faster)
- Final performance (should be similar or better)

### Visual Testing
With fixed friction:
- Behavior should be **more consistent** across episodes
- Easier to judge if gliding is happening
- Clearer cause-effect from weight changes

## Documentation Updates Needed

Files mentioning "friction (±20%)" in domain randomization:
- ✅ `tensegrity_mjc_simulation.py` - Updated docstring
- ⏸️ `gpu_pretraining_SAC.py` - Print statements (informational only, no code change needed)
- ⏸️ `DOMAIN_RANDOMIZATION_IMPLEMENTATION.md` - Reference doc (update later)

## Rollback

To re-enable friction randomization:

```powershell
# Option 1: Manual edit
# Uncomment lines 176-186 in tensegrity_mjc_simulation.py

# Option 2: Git
git diff HEAD -- mujoco_physics_engine/tensegrity_mjc_simulation.py
# Review and selectively revert
```

---

**Status**: ✅ Friction randomization disabled, ready for clean reward weight testing
