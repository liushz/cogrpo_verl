# Code Review Skill

Automated code review for ML/RL training code.

## Focus Areas

### 1. Performance
- GPU utilization efficiency
- Memory allocation patterns
- Data loading bottlenecks
- Distributed training overhead

### 2. Correctness
- Gradient computation
- Loss function implementation
- Reward calculation
- Tokenization handling

### 3. Safety
- CUDA error handling
- Checkpoint recovery
- Gradient clipping
- NaN detection

## Review Checklist

```markdown
- [ ] No hardcoded paths
- [ ] Proper error handling for CUDA OOM
- [ ] Distributed training safe (DDP/FSDP)
- [ ] Efficient data loading (prefetch, pin_memory)
- [ ] Proper gradient scaling for mixed precision
- [ ] Wandb/tensorboard logging configured
- [ ] Checkpoint saving works correctly
```

## Usage

```
/review <file_path>
/review-diff <base_branch>
```

## Output Format

```
## Summary
<brief summary>

## Critical Issues
- <issue with line number>

## Suggestions
- <improvement suggestion>

## Performance Notes
- <performance observations>
```
