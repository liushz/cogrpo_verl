# LLM Training Skill

Specialized skill for LLM training tasks in this codebase.

## Context

This project uses:
- **verl** framework for RL training
- **vLLM** for rollout generation
- **GRPO** (Group Relative Policy Optimization)
- **CompassVerifier** for answer verification

## Common Tasks

### 1. Debug Training Issues

```
/train-debug
```
Analyzes:
- GPU memory usage
- Distributed training errors
- Gradient flow issues
- Data loading bottlenecks

### 2. Check Cluster Status

```
/cluster-status
```
Shows:
- GPU availability
- Running jobs
- Queue status

### 3. Analyze Training Logs

```
/log-analyze <log_file>
```
Extracts:
- Loss curves
- Learning rate schedule
- Gradient norms
- Token throughput

## Key Files

- `scripts/eval_co_grpo_*.py` - Evaluation scripts
- `verl/workers/rollout/` - Rollout workers
- `scripts/verifier_data_gen/` - Data generation

## Best Practices

1. Always check GPU memory before large batch training
2. Use `torch.cuda.empty_cache()` between experiments
3. Monitor wandb for real-time metrics
4. Save checkpoints frequently
