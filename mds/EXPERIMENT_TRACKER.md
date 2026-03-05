# 32-GPU CoGRPO Experiment Tracker

**Last Updated:** 2026-02-26 01:28

## Current Status Summary

### Inqueue/Starting (4 experiments)

| Job Name | Showname | Status | Notes |
|----------|----------|--------|-------|
| rjob-cogrpo32-dec-int1-e1k2-0226fix-65504569-61f6b | rjob_cogrpo32_dec_int1_e1k2_0226fix | Inqueue | max_interventions=1, cf_k=2 |
| rjob-cogrpo32-hf170-lora1705-dapo17k-pl2048-45cf0 | rjob_cogrpo32_hf170_lora1705_dapo17k_pl2048_cfK2E2_int2_32k_v4 | Inqueue | hf-170 + lora-1705 |
| rjob-cogrpo32-hf181-nolora-dapo17k-pl2048-cf-eb6ab | rjob_cogrpo32_hf181_nolora_dapo17k_pl2048_cfK2E2_int2_32k_v4 | Inqueue | hf-181 no lora, full verifier |
| rjob-cogrpo32-hf170-lora1705-dapo17k-pl2048-ac2fd | rjob_cogrpo32_hf170_lora1705_dapo17k_pl2048_cfK2E2_int2_32k_exp16_v3_fixkey | Inqueue | fixkey version |

### Failed (6+ experiments)

| Job Name | Error Type | Root Cause |
|----------|------------|------------|
| rjob_cogrpo32_hf170_lora1705_dapo17k_pl2048_cfK2E2_int2_32k_v3 | KeyError | 'layers.0.self_attn.qkv_proj.weight' - weight key mismatch |
| rjob_cogrpo32_dec_int1_e1k2_02251945 | KeyError | 'layers.0.self_attn.qkv_proj.weight' - weight key mismatch |
| rjob_cogrpo32_hf181_nolora_dapo17k_pl2048_cfK2E2_int2_32k_v3 | NameError | 'get_config_value' is not defined |
| rjob_cogrpo32_cfbranch_k4_e2_bsz64_v3 | - | - |
| rjob_cogrpo32_hf170_lora1705_cfK2E2_int2_32k | KeyError | 'layers.0.self_attn.qkv_proj.weight' |

## Known Issues

### 1. KeyError: 'layers.0.self_attn.qkv_proj.weight' (CRITICAL)
- **Affected experiments:** ALL hf-170 based experiments
- **Root Cause Analysis:**
  - vLLM's Qwen2 model expects **merged** qkv weights: `layers.X.self_attn.qkv_proj.weight`
  - FSDP state dict has **separated** weights: `layers.X.self_attn.q_proj.weight`, `k_proj.weight`, `v_proj.weight`
  - vLLM's `load_weights()` has logic to convert q_proj->qkv_proj, but the key prefix mismatch causes the error
  - The error occurs because `params_dict` uses a different prefix format than the incoming weights
- **Call Stack:**
  ```
  fsdp_vllm.py:253 __enter__ -> update_params()
  fsdp_vllm.py:358 update_params -> model.load_weights()
  vllm/qwen2.py:416 load_weights -> params_dict[name] <- KeyError
  ```
- **Location:** `verl/workers/sharding_manager/fsdp_vllm.py:358`
- **Possible Fix:** Check `convert_weight_keys()` in `verl/utils/model.py` for proper key prefix handling
- **Status:** BLOCKING - Needs weight key remapping or model format conversion

### 2. NameError: 'get_config_value' is not defined
- **Affected experiments:** hf-181 nolora experiments
- **Location:** `verl/trainer/ppo/ray_trainer.py:3945`
- **Status:** Code bug, needs fix

## Experiment Configuration Details

### Common Settings (32-GPU)
- **Nodes:** 4
- **GPUs per node:** 8
- **Total GPUs:** 32
- **Train batch size:** 128
- **Micro batch size per GPU:** 2
- **Response length:** 32k (32768)
- **Rollout n:** 8
- **CF branch k:** 2
- **Max interventions:** 1-2

### Model Variants
1. **hf-170 + lora-1705**
   - Actor: cispo-cold-start-model/hf-170
   - Verifier LoRA: checkpoint-1705

2. **hf-181 nolora**
   - Actor: cold_start_full_qwen2d5-7b/hf-181
   - Verifier: Full model update (no LoRA)

## Action Items
- [ ] Fix KeyError for weight loading
- [ ] Fix NameError for get_config_value
- [ ] Monitor starting experiments until 10+ steps
- [ ] Debug and restart failed experiments

## Log Locations
- Main logs: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/logs/`
- SwanLab: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/swanlab/`
- Checkpoints: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/checkpoints/`
