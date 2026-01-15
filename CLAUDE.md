# Claude Code Guidelines for dgx-spark-setup

## Best Practices

### Memory/Batch Size Adjustments
- When OOM occurs, reduce batch size by **half (2x)**, not more
- Reducing by 4x is overly conservative and unnecessarily slows training
- Example: if batch=8 crashes, try batch=4 (not batch=2)

### Testing Changes
- Training scripts must be tested on actual DGX Spark hardware (not Mac/x86)
- Verify scripts run to at least the first training step before merging
- Check that loss values are reasonable (not NaN/inf)

### Documentation
- Keep README.md table of contents updated when adding sections
- Add new scripts/docs to the contents table with short descriptions
