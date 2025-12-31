# Combined Gather Optimization - Phase 2.3

## Status
- **Branch**: `feature/mpi-gather-optimization`
- **Current State**: Production-ready code (debug prints, tracemalloc removed)
- **Next Step**: Implement combined gather optimization

## Goal
Instead of doing 3 separate gathers (bundles, features, errors), gather bundles once and compute features/errors on root. This reduces MPI communication from 3 operations to 1.

## Current Implementation
```python
# In _master_iteration():
bundles_sim = self.comm_manager.concatenate_array_at_root_fast(local_pricing_results, root=0)
x_sim = self.feature_manager.compute_gathered_features(local_pricing_results, timing_dict=None)
errors_sim = self.feature_manager.compute_gathered_errors(local_pricing_results, timing_dict=None)
```

**Problem**: `compute_gathered_features` and `compute_gathered_errors` each do their own gather internally, so we're doing 3 separate gathers.

## Proposed Optimization
```python
# Gather bundles once
bundles_sim = self.comm_manager.concatenate_array_at_root_fast(local_pricing_results, root=0)

# Compute features/errors on root from gathered bundles (no additional gather)
if self.is_root():
    x_sim = self.feature_manager.compute_rank_features(bundles_sim)  # Compute on root
    errors_sim = (self.data_manager.input_data["errors"] * bundles_sim).sum(1)  # Compute on root
else:
    x_sim = None
    errors_sim = None
```

## Key Considerations

### 1. Feature Computation
- **Current**: Each rank computes features locally, then gathers
- **Proposed**: Root computes features from gathered bundles
- **Requirement**: `compute_rank_features()` must work with the full gathered bundle array
- **Check**: Does `compute_rank_features()` handle the case where `num_local_agents` doesn't match the bundle array length?

### 2. Error Computation
- **Current**: Each rank computes errors locally, then gathers
- **Proposed**: Root computes errors from `input_data["errors"]` and gathered bundles
- **Requirement**: `input_data["errors"]` must be available on root with shape `(num_simulations * num_agents, num_items)`
- **Check**: Is `input_data` available on root? What's the shape of errors?

### 3. Data Availability
- **Bundles**: Gathered to root ✓
- **Errors tensor**: Need to verify `input_data["errors"]` is available on root
- **Agent/item data**: Need to verify `input_data` or `local_data` has what's needed for feature computation

### 4. Implementation Details
- **Location**: `bundlechoice/estimation/row_generation.py`, `_master_iteration()` method
- **Timing**: Still track `mpi_gather` as single operation
- **Backward compatibility**: Should work the same, just more efficient

## Testing Strategy

1. **Local test** (2-4 ranks):
   - Verify features/errors computed correctly on root
   - Compare results with old method (should be identical)
   - Check timing improvement

2. **HPC test** (160-400 ranks):
   - Measure actual MPI gather time reduction
   - Verify correctness at scale
   - Compare total runtime

## Files to Modify

1. `bundlechoice/estimation/row_generation.py`:
   - Modify `_master_iteration()` to gather bundles once
   - Compute features/errors on root from gathered bundles
   - Remove calls to `compute_gathered_features()` and `compute_gathered_errors()`

2. **Optional**: `bundlechoice/feature_manager.py`:
   - May need to add method to compute features from full array (not just local)
   - Or verify `compute_rank_features()` works with full array

## Questions to Answer

1. Does `compute_rank_features()` work with full gathered bundle array?
2. Is `input_data["errors"]` available on root with correct shape?
3. Does feature computation need agent/item data that's only in `local_data`?
4. What's the shape of `bundles_sim` after gather? `(num_simulations * num_agents, num_items)`?

## Expected Benefits

- **Reduce MPI operations**: 3 gathers → 1 gather
- **Reduce synchronization overhead**: Fewer MPI barriers
- **Potential speedup**: 2-3x faster gather phase (depending on network)

## Implementation Notes

- Keep the early convergence check (it's already optimized)
- Maintain backward compatibility with timing dict
- Add comments explaining the optimization
- Test thoroughly - this changes the computation flow

## Next Steps

1. Review this document
2. Implement combined gather in `_master_iteration()`
3. Test locally (2-4 ranks)
4. Push to branch
5. HPC agent will test at scale

