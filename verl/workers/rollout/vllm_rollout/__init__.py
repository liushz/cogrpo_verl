# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


# ============================================================================
# vLLM Qwen3 Fix: Patch Qwen3Attention to handle HF weight names
# This fixes the KeyError: 'layers.0.self_attn.qkv_proj.weight' issue
# ============================================================================
def _patch_vllm_qwen3():
    """Patch vLLM Qwen3Attention to add load_weights method for HF compatibility."""
    try:
        from vllm.model_executor.models import qwen3 as qwen3_module
        from vllm.model_executor.models.utils import default_weight_loader

        Qwen3Attention = qwen3_module.Qwen3Attention

        # Check if load_weights already exists
        if not hasattr(Qwen3Attention, 'load_weights'):
            def load_weights(self, weights):
                """Load weights mapping HF q_proj/k_proj/v_proj to vLLM qkv_proj"""
                stacked_params_mapping = [
                    ('qkv_proj', 'q_proj', 'q'),
                    ('qkv_proj', 'k_proj', 'k'),
                    ('qkv_proj', 'v_proj', 'v'),
                ]
                params_dict = dict(self.named_parameters(remove_duplicate=False))
                loaded_params = set()

                for name, loaded_weight in weights:
                    if 'rotary_emb.inv_freq' in name:
                        continue
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        if weight_name not in name:
                            continue
                        vllm_name = name.replace(weight_name, param_name)
                        if vllm_name not in params_dict:
                            continue
                        param = params_dict[vllm_name]
                        weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(vllm_name)
                        break
                return loaded_params

            # Add the method to the class
            Qwen3Attention.load_weights = load_weights
            print("[VERL PATCH] vLLM Qwen3Attention load_weights patch applied successfully!")
    except Exception:
        # Silently ignore if vllm or qwen3 is not available
        pass


# Apply the patch when this module is imported
_patch_vllm_qwen3()
# ============================================================================


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


vllm_package_name = "vllm"
vllm_package_version = get_version(vllm_package_name)
if vllm_package_version is None:
    raise PackageNotFoundError("To use vllm rollout, please ensure the 'vllm' package is properly installed. See https://verl.readthedocs.io/en/latest/start/install.html for more details")

###
# package_version = get_version(package_name)
# [SUPPORT AMD:]
# Do not call any torch.cuda* API here, or ray actor creation import class will fail.
if "ROCM_PATH" in os.environ:
    import re

    match = re.match(r"(\d+\.\d+\.?\d*)", vllm_package_version)
    if match:
        vllm_package_version = match.group(1)
    else:
        raise ValueError(f"Warning: Could not parse version format: {vllm_package_version}")
###

if Version(vllm_package_version) <= Version("0.6.3"):
    vllm_mode = "customized"
    from .fire_vllm_rollout import FIREvLLMRollout  # noqa: F401
    from .vllm_rollout import vLLMRollout  # noqa: F401
else:
    vllm_mode = "spmd"
    from .vllm_rollout_spmd import vLLMAsyncRollout, vLLMRollout  # noqa: F401
