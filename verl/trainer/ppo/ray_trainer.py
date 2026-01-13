# type: ignore
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import sys
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type
import math
from functools import reduce

import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import einops

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.debug.performance import _timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean, pad_sequence_to_length
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto, 
    adv_estimator, 
    gamma=1.0, 
    lam=1.0, 
    num_repeat=1, 
    multi_turn=False, 
    norm_adv_by_std_in_grpo=True, 
    config=None,
    training_progress=0.0,
):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=torch.tensor(gamma) if not isinstance(gamma, torch.Tensor) else gamma,
            lam=torch.tensor(lam) if not isinstance(lam, torch.Tensor) else lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.REPRO_PPO:
        advantages, returns = core_algos.compute_repro_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            token_level_repro_rewards=data.batch["token_level_repro_rewards"],
            response_mask=data.batch["response_mask"],
            repro_mask=data.batch["repro_mask"],
            index=data.non_tensor_batch["uid"],
            gamma=torch.tensor(gamma) if not isinstance(gamma, torch.Tensor) else gamma,
            lam=torch.tensor(lam) if not isinstance(lam, torch.Tensor) else lam,
            config=config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        response_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = response_mask.size(1)
            # This mask is the one intended for GRPO
            response_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=response_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REPRO_GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        
        advantages, returns = core_algos.compute_repro_grpo_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            token_level_repro_rewards=data.batch["token_level_repro_rewards"],
            response_mask=grpo_calculation_mask,
            repro_mask=data.batch["repro_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REPRO_REINFORCE_PLUS_PLUS:
        response_mask = data.batch["response_mask"]
        advantages, returns = core_algos.compute_repro_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            token_level_repro_rewards=data.batch["token_level_repro_rewards"],
            response_mask=response_mask,
            repro_mask=data.batch["repro_mask"],
            config=config, 
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REPRO_REINFORCE_PLUS_PLUS_BASELINE:
        response_mask = data.batch["response_mask"]
        advantages, returns = core_algos.compute_repro_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            token_level_repro_rewards=data.batch["token_level_repro_rewards"],
            response_mask=response_mask,
            repro_mask=data.batch["repro_mask"],
            index=data.non_tensor_batch["uid"],
            config=config, 
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.CO_GRPO:
        # Co-GRPO: Dual-stream advantage computation
        # Control stream mask
        control_response_mask = data.batch.get("control_response_mask", data.batch["response_mask"])
        exp_response_mask = data.batch.get("exp_response_mask", data.batch["response_mask"])
        
        # Get control and experimental rewards
        control_token_level_rewards = data.batch.get("control_token_level_rewards", None)
        exp_token_level_rewards = data.batch.get("exp_token_level_rewards", data.batch["token_level_rewards"])
        
        # Get verifier rewards (relative advantage: r_B - r_A)
        verifier_token_level_rewards = data.batch.get("verifier_token_level_rewards", None)
        verifier_response_mask = data.batch.get("verifier_response_mask", None)
        
        if config is None:
            raise ValueError("config is required for CO_GRPO. Make sure self.config.algorithm is properly initialized.")
        
        # Convert OmegaConf to dict if needed for compatibility
        if hasattr(config, 'get'):
            control_group_weight = config.get("control_group_weight", 0.5)
        else:
            # If config is a dict-like object, use getattr or direct access
            control_group_weight = getattr(config, "control_group_weight", 0.5) if hasattr(config, "control_group_weight") else 0.5
        
        policy_advantages, verifier_advantages, policy_returns, verifier_returns, metadata = core_algos.compute_co_grpo_advantage(
            token_level_rewards=exp_token_level_rewards,
            response_mask=exp_response_mask,
            index=data.non_tensor_batch["uid"],
            control_token_level_rewards=control_token_level_rewards,
            control_response_mask=control_response_mask,
            verifier_token_level_rewards=verifier_token_level_rewards,
            verifier_response_mask=verifier_response_mask,
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            control_group_weight=control_group_weight,
            config=config,
            training_progress=training_progress,
        )
        
        # Store metadata for logging
        data.meta_info['co_grpo_metadata'] = metadata
        
        data.batch["advantages"] = policy_advantages
        data.batch["returns"] = policy_returns
        
        # Store verifier advantages separately if available
        if verifier_advantages is not None:
            data.batch["verifier_advantages"] = verifier_advantages
            data.batch["verifier_returns"] = verifier_returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GAE,
            AdvantageEstimator.REPRO_PPO,
        ]:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.REPRO_GRPO,
            AdvantageEstimator.REPRO_REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REPRO_REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.CO_GRPO,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "reward_model"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    if key == "predicted_answer":
                        continue
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            # For Co-GRPO, pass verifier config to actor_rollout_ref worker
            # The worker needs to know about verifier LoRA configuration
            from omegaconf import OmegaConf, DictConfig
            actor_rollout_ref_config = OmegaConf.to_container(self.config.actor_rollout_ref, resolve=True)
            actor_rollout_ref_config = DictConfig(actor_rollout_ref_config)
            # Disable struct mode to allow adding verifier key
            OmegaConf.set_struct(actor_rollout_ref_config, False)
            if hasattr(self.config, 'verifier'):
                actor_rollout_ref_config.verifier = self.config.verifier
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=actor_rollout_ref_config,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        BaseCheckpointManager.local_mkdir(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        # For Co-GRPO, use exp_attention_mask; for regular, use attention_mask
        if "exp_attention_mask" in batch.batch:
            attention_mask = batch.batch["exp_attention_mask"]
        else:
            attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # Check if using Co-GRPO (dual-stream rollout)
                        use_co_grpo = self.config.algorithm.adv_estimator == AdvantageEstimator.CO_GRPO
                        
                        if use_co_grpo:
                            # For Co-GRPO, repeat gen_batch first so each prompt generates multiple pairs
                            # e.g., rollout.n=8 → each prompt repeated 8 times → generates 8 pairs (16 samples)
                            # #region agent log
                            with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "A,B,C,D,E",
                                    "location": "ray_trainer.py:1092",
                                    "message": "Before gen_batch.repeat: original batch size",
                                    "data": {
                                        "gen_batch_batch_size": str(gen_batch.batch.batch_size),
                                        "rollout_n": self.config.actor_rollout_ref.rollout.n
                                    },
                                    "timestamp": int(time.time() * 1000)
                                }) + "\n")
                            # #endregion
                            gen_batch_repeated = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            # #region agent log
                            with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "A,B,C,D,E",
                                    "location": "ray_trainer.py:1093",
                                    "message": "After gen_batch.repeat: gen_batch_repeated size",
                                    "data": {
                                        "gen_batch_repeated_batch_size": str(gen_batch_repeated.batch.batch_size),
                                        "rollout_n": self.config.actor_rollout_ref.rollout.n
                                    },
                                    "timestamp": int(time.time() * 1000)
                                }) + "\n")
                            # #endregion
                            # Use dual-stream rollout for Co-GRPO
                            if not self.async_rollout_mode:
                                # Get intervention mode from config
                                intervention_mode = self.config.algorithm.get("verifier_intervention_mode", "by_response")
                                token_check_interval = self.config.algorithm.get("token_check_interval", 5)
                                entropy_threshold = self.config.algorithm.get("entropy_threshold", 0.5)
                                use_entropy_filter = self.config.algorithm.get("use_entropy_filter", True)
                                max_interventions = self.config.algorithm.get("max_interventions", 3)
                                confidence_threshold = self.config.algorithm.get("confidence_threshold", 0.7)
                                
                                # Call dual_stream_rollout (tokenizer is stored in rollout worker)
                                gen_batch_output = self.actor_rollout_wg.dual_stream_rollout(
                                    gen_batch_repeated,
                                    intervention_mode=intervention_mode,
                                    token_check_interval=token_check_interval,
                                    entropy_threshold=entropy_threshold,
                                    use_entropy_filter=use_entropy_filter,
                                    max_interventions=max_interventions,
                                    confidence_threshold=confidence_threshold,
                                )
                                # #region agent log
                                with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "A,B,C,D,E",
                                        "location": "ray_trainer.py:1113",
                                        "message": "After dual_stream_rollout: gen_batch_output size",
                                        "data": {
                                            "gen_batch_output_batch_size": str(gen_batch_output.batch.batch_size),
                                            "gen_batch_repeated_batch_size": str(gen_batch_repeated.batch.batch_size),
                                            "rollout_n": self.config.actor_rollout_ref.rollout.n
                                        },
                                        "timestamp": int(time.time() * 1000)
                                    }) + "\n")
                                # #endregion
                            else:
                                # For async mode, we need to handle differently
                                # For now, fall back to regular generation
                                logger.warning("Co-GRPO dual-stream rollout not yet supported in async mode, using regular generation")
                                self.async_rollout_manager.wake_up()
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_repeated)
                                self.async_rollout_manager.sleep()
                        else:
                            # Regular rollout
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                self.async_rollout_manager.wake_up()
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                self.async_rollout_manager.sleep()
                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    # For both regular GRPO and Co-GRPO, we repeat the batch
                    # - Regular GRPO: repeat n times, each generates 1 response → n responses per prompt
                    # - Co-GRPO: repeat n times, each generates 1 pair (control+exp) → n pairs (2n samples) per prompt
                    # #region agent log
                    with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A,B,C,D,E",
                            "location": "ray_trainer.py:1153",
                            "message": "Before batch.repeat: batch size and gen_batch_output size",
                            "data": {
                                "batch_batch_size": str(batch.batch.batch_size),
                                "gen_batch_output_batch_size": str(gen_batch_output.batch.batch_size),
                                "rollout_n": self.config.actor_rollout_ref.rollout.n,
                                "use_co_grpo": use_co_grpo,
                                "batch_keys": list(batch.batch.keys()),
                                "gen_batch_output_keys": list(gen_batch_output.batch.keys())
                            },
                            "timestamp": int(time.time() * 1000)
                        }) + "\n")
                    # #endregion
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # #region agent log
                    with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A,B,C,D,E",
                            "location": "ray_trainer.py:1154",
                            "message": "After batch.repeat: batch size before union",
                            "data": {
                                "batch_batch_size": str(batch.batch.batch_size),
                                "gen_batch_output_batch_size": str(gen_batch_output.batch.batch_size),
                                "rollout_n": self.config.actor_rollout_ref.rollout.n
                            },
                            "timestamp": int(time.time() * 1000)
                        }) + "\n")
                    # #endregion
                    batch = batch.union(gen_batch_output)

                    # For Co-GRPO, handle response masks for both streams
                    if use_co_grpo:
                        # Check if masks already exist from rollout (to avoid recomputation)
                        if "control_response_mask" not in batch.batch or "exp_response_mask" not in batch.batch:
                            # Compute response masks for both streams
                            from verl.utils.torch_functional import get_response_mask
                            eos_token_id = batch.meta_info["eos_token_id"]
                            batch.batch["control_response_mask"] = get_response_mask(
                                response_id=batch.batch["control_responses"],
                                eos_token=eos_token_id,
                                dtype=batch.batch["control_attention_mask"].dtype
                            )
                            batch.batch["exp_response_mask"] = get_response_mask(
                                response_id=batch.batch["exp_responses"],
                                eos_token=eos_token_id,
                                dtype=batch.batch["exp_attention_mask"].dtype
                            )
                        
                        # For by_step mode, use exp_loss_mask if available (to exclude Hints from loss)
                        # Otherwise, use exp_response_mask for by_response mode
                        exp_resp_len = batch.batch["exp_responses"].size(1)
                        if "exp_loss_mask" in batch.batch:
                            # exp_loss_mask is full sequence mask; clip to response segment
                            batch.batch["response_mask"] = batch.batch["exp_loss_mask"][:, -exp_resp_len:]
                        else:
                            # exp_response_mask already matches response length; keep a safe slice
                            batch.batch["response_mask"] = batch.batch["exp_response_mask"][:, -exp_resp_len:]
                    else:
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    # For Co-GRPO, use exp_attention_mask; for regular, use attention_mask
                    if use_co_grpo:
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["exp_attention_mask"], dim=-1).tolist()
                    else:
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # For Co-GRPO, compute rewards for both control and experimental streams
                        if use_co_grpo:
                            # Compute rewards for control stream
                            control_batch = batch.select(
                                batch_keys=["control_input_ids", "control_attention_mask", "control_position_ids"],
                                # Preserve all non-tensor fields (reward_model, data_source, etc.) for reward_fn
                                non_tensor_batch_keys=list(batch.non_tensor_batch.keys())
                            )
                            control_batch.batch["input_ids"] = control_batch.batch.pop("control_input_ids")
                            control_batch.batch["attention_mask"] = control_batch.batch.pop("control_attention_mask")
                            control_batch.batch["position_ids"] = control_batch.batch.pop("control_position_ids")
                            control_batch.batch["responses"] = batch.batch["control_responses"]
                            # Reward fn expects raw prompts to compute valid prompt/response lengths
                            if "prompts" in batch.batch.keys():
                                control_batch.batch["prompts"] = batch.batch["prompts"]
                            else:
                                control_prompt_len = control_batch.batch["input_ids"].size(1) - batch.batch["control_responses"].size(1)
                                control_batch.batch["prompts"] = control_batch.batch["input_ids"][..., :control_prompt_len]
                            
                            # Compute rewards for experimental stream
                            exp_batch = batch.select(
                                batch_keys=["exp_input_ids", "exp_attention_mask", "exp_position_ids"],
                                non_tensor_batch_keys=list(batch.non_tensor_batch.keys())
                            )
                            exp_batch.batch["input_ids"] = exp_batch.batch.pop("exp_input_ids")
                            exp_batch.batch["attention_mask"] = exp_batch.batch.pop("exp_attention_mask")
                            exp_batch.batch["position_ids"] = exp_batch.batch.pop("exp_position_ids")
                            exp_batch.batch["responses"] = batch.batch["exp_responses"]
                            if "exp_prompts" in batch.batch.keys():
                                exp_batch.batch["prompts"] = batch.batch["exp_prompts"]
                            else:
                                exp_prompt_len = exp_batch.batch["input_ids"].size(1) - batch.batch["exp_responses"].size(1)
                                exp_batch.batch["prompts"] = exp_batch.batch["input_ids"][..., :exp_prompt_len]
                            
                            if self.config.reward_model.launch_reward_fn_async:
                                tokenizer_name_or_path = self.tokenizer.name_or_path
                                future_control_reward = compute_reward_async.remote(control_batch, self.config, tokenizer_name_or_path)
                                future_exp_reward = compute_reward_async.remote(exp_batch, self.config, tokenizer_name_or_path)
                                control_reward_tensor, _ = ray.get(future_control_reward)
                                exp_reward_tensor, _ = ray.get(future_exp_reward)
                            else:
                                control_reward_tensor, _ = compute_reward(control_batch, self.reward_fn)
                                exp_reward_tensor, _ = compute_reward(exp_batch, self.reward_fn)

                            # ========== Co-GRPO Diagnostic: Batch Comparison ==========
                            print(f"\n{'='*80}", file=sys.stderr)
                            print(f"[Co-GRPO Batch Statistics] After Reward Computation", file=sys.stderr)
                            print(f"{'='*80}", file=sys.stderr)

                            # Print sample comparison
                            sample_idx = 0
                            control_resp = self.tokenizer.decode(
                                batch.batch['control_responses'][sample_idx][:batch.batch['control_attention_mask'][sample_idx].sum()],
                                skip_special_tokens=True
                            )
                            exp_resp = self.tokenizer.decode(
                                batch.batch['exp_responses'][sample_idx][:batch.batch['exp_attention_mask'][sample_idx].sum()],
                                skip_special_tokens=True
                            )

                            print(f"\n[Control Sample {sample_idx}] (first 500 chars):\n{control_resp[:500]}...\n", file=sys.stderr)
                            print(f"[Exp Sample {sample_idx}] (first 500 chars):\n{exp_resp[:500]}...\n", file=sys.stderr)

                            # Intervention stats
                            if 'num_interventions' in batch.non_tensor_batch:
                                interventions = batch.non_tensor_batch['num_interventions']
                                print(f"[Intervention Stats]:", file=sys.stderr)
                                print(f"  Mean: {float(np.mean(interventions)):.2f}", file=sys.stderr)
                                print(f"  Max: {int(np.max(interventions))}", file=sys.stderr)
                                print(f"  Min: {int(np.min(interventions))}", file=sys.stderr)

                            # Reward comparison
                            control_outcome_early = control_reward_tensor.squeeze() if control_reward_tensor.dim() > 1 else control_reward_tensor
                            exp_outcome_early = exp_reward_tensor.squeeze() if exp_reward_tensor.dim() > 1 else exp_reward_tensor

                            print(f"[Reward Comparison]:", file=sys.stderr)
                            print(f"  Control reward: {float(control_outcome_early.mean()):.4f} ± {float(control_outcome_early.std()):.4f}", file=sys.stderr)
                            print(f"  Exp reward: {float(exp_outcome_early.mean()):.4f} ± {float(exp_outcome_early.std()):.4f}", file=sys.stderr)
                            print(f"  Gap (exp - control): {float(exp_outcome_early.mean() - control_outcome_early.mean()):.4f}", file=sys.stderr)
                            print(f"{'='*80}\n", file=sys.stderr)
                            # ========== End Co-GRPO Diagnostic ==========

                            # Ensure rewards are token-level (expand to match response length)
                            control_resp_len = batch.batch["control_responses"].size(1)
                            exp_resp_len = batch.batch["exp_responses"].size(1)
                            
                            # Handle different reward tensor shapes
                            if control_reward_tensor.dim() == 1:
                                # Outcome rewards: expand to token level
                                control_token_rewards = control_reward_tensor.unsqueeze(-1).expand(-1, control_resp_len)
                            else:
                                # Already token-level
                                control_token_rewards = control_reward_tensor
                            
                            if exp_reward_tensor.dim() == 1:
                                # Outcome rewards: expand to token level
                                exp_token_rewards = exp_reward_tensor.unsqueeze(-1).expand(-1, exp_resp_len)
                            else:
                                # Already token-level
                                exp_token_rewards = exp_reward_tensor
                            
                            # Store rewards in batch
                            batch.batch["control_token_level_rewards"] = control_token_rewards
                            batch.batch["exp_token_level_rewards"] = exp_token_rewards
                            
                            # Compute verifier relative rewards: r_B - r_A (outcome level)
                            from verl.trainer.ppo.reward import compute_relative_advantage_reward
                            
                            def get_outcome_reward(token_rewards):
                                """Convert token-level rewards to outcome rewards if needed."""
                                if token_rewards.dim() == 1:
                                    # Already outcome-level
                                    return token_rewards
                                elif token_rewards.size(-1) == 1:
                                    # Shape [batch, 1], squeeze it
                                    return token_rewards.squeeze(-1)
                                else:
                                    # Shape [batch, seq_len], sum it
                                    return token_rewards.sum(dim=-1)
                            
                            control_outcome = get_outcome_reward(control_token_rewards)
                            exp_outcome = get_outcome_reward(exp_token_rewards)
                            verifier_outcome_rewards = compute_relative_advantage_reward(
                                control_rewards=control_outcome,
                                exp_rewards=exp_outcome
                            )
                            
                            # Apply intervention penalty to prevent over-intervention
                            if hasattr(self.config.algorithm, 'intervention_penalty'):
                                penalty_config = self.config.algorithm.intervention_penalty
                                lambda_freq = penalty_config.get('freq_coef', 0.0)
                                lambda_len = penalty_config.get('len_coef', 0.0)
                                
                                if lambda_freq > 0 or lambda_len > 0:
                                    # Get intervention statistics from non_tensor_batch
                                    num_interventions = batch.non_tensor_batch.get('num_interventions', None)
                                    hint_token_counts = batch.non_tensor_batch.get('hint_token_counts', None)
                                    
                                    if num_interventions is not None and hint_token_counts is not None:
                                        # Compute intervention cost
                                        intervention_costs = torch.zeros_like(verifier_outcome_rewards)
                                        for i in range(len(verifier_outcome_rewards)):
                                            cost = lambda_freq * num_interventions[i] + lambda_len * hint_token_counts[i] / 100.0  # Normalize token count
                                            intervention_costs[i] = cost
                                        
                                        # Subtract cost from verifier rewards
                                        verifier_outcome_rewards = verifier_outcome_rewards - intervention_costs
                                        
                                        # Log intervention statistics
                                        metrics['co_grpo/avg_interventions'] = np.mean(num_interventions)
                                        metrics['co_grpo/avg_hint_tokens'] = np.mean(hint_token_counts)
                                        metrics['co_grpo/avg_intervention_cost'] = intervention_costs.mean().item()
                                        metrics['co_grpo/max_interventions'] = np.max(num_interventions)
                                        metrics['co_grpo/min_interventions'] = np.min(num_interventions)
                            
                            # Log reward comparison statistics
                            metrics['co_grpo/control_reward_mean'] = control_outcome.mean().item()
                            metrics['co_grpo/control_reward_std'] = control_outcome.std().item()
                            metrics['co_grpo/exp_reward_mean'] = exp_outcome.mean().item()
                            metrics['co_grpo/exp_reward_std'] = exp_outcome.std().item()
                            
                            # Log relative reward statistics (before penalty)
                            raw_relative_rewards = exp_outcome - control_outcome
                            metrics['co_grpo/relative_reward_mean'] = raw_relative_rewards.mean().item()
                            metrics['co_grpo/relative_reward_std'] = raw_relative_rewards.std().item()
                            metrics['co_grpo/relative_reward_positive_ratio'] = (raw_relative_rewards > 0).float().mean().item()
                            
                            # Log verifier reward after penalty
                            metrics['co_grpo/verifier_reward_mean'] = verifier_outcome_rewards.mean().item()
                            metrics['co_grpo/verifier_reward_std'] = verifier_outcome_rewards.std().item()
                            
                            # Log improvement ratio (how much exp improves over control)
                            improvement_ratio = (exp_outcome - control_outcome) / (control_outcome.abs() + 1e-6)
                            metrics['co_grpo/improvement_ratio_mean'] = improvement_ratio.mean().item()
                            
                            # Expand to token level for verifier
                            batch.batch["verifier_token_level_rewards"] = verifier_outcome_rewards.unsqueeze(-1).expand(-1, exp_resp_len)
                            batch.batch["verifier_response_mask"] = batch.batch["exp_response_mask"]
                            
                            # Use experimental rewards as main rewards for policy training
                            batch.batch["token_level_rewards"] = exp_token_rewards
                            reward_tensor = exp_token_rewards
                            reward_extra_infos_dict = {}
                        else:
                            # Regular reward computation
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                # Pass tokenizer name/path instead of object to avoid serialization issues
                                tokenizer_name_or_path = self.tokenizer.name_or_path
                                future_reward = compute_reward_async.remote(batch, self.config, tokenizer_name_or_path)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs 
                    with _timer("old_log_prob", timing_raw):
                        # For Co-GRPO, map exp_* fields to standard fields for compute_log_prob
                        if use_co_grpo:
                            # Create a temporary batch with standard field names for compute_log_prob
                            log_prob_batch = batch.select(
                                batch_keys=["exp_input_ids", "exp_attention_mask", "exp_position_ids"],
                                non_tensor_batch_keys=["uid"]
                            )
                            log_prob_batch.batch["input_ids"] = log_prob_batch.batch.pop("exp_input_ids")
                            log_prob_batch.batch["attention_mask"] = log_prob_batch.batch.pop("exp_attention_mask")
                            log_prob_batch.batch["position_ids"] = log_prob_batch.batch.pop("exp_position_ids")
                            log_prob_batch.batch["responses"] = batch.batch["exp_responses"]
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(log_prob_batch)
                        else:
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.config.algorithm.adv_estimator in [
                        AdvantageEstimator.REPRO_PPO,
                        AdvantageEstimator.REPRO_GRPO,
                        AdvantageEstimator.REPRO_REINFORCE_PLUS_PLUS,
                        AdvantageEstimator.REPRO_REINFORCE_PLUS_PLUS_BASELINE
                    ]:
                        with _timer("repro_score", timing_raw):
                            batch_size, seq_len = batch.batch["input_ids"].shape
                            prompt_len = batch.batch["prompts"].size(1)
                            response_len = batch.batch["responses"].size(1)
                            valid_prompt_len = batch.batch["attention_mask"][:, :prompt_len].sum(1)
                            valid_response_len = batch.batch["attention_mask"][:, prompt_len:].sum(1)
                            device = batch.batch.device

                            end_think_token_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
                            think_mask = ~(batch.batch["responses"] == end_think_token_id).cumsum(1).bool()

                            line_break_id = (
                                self.tokenizer.encode(".\n", add_special_tokens=False)[0],
                                self.tokenizer.encode("?\n", add_special_tokens=False)[0],
                                self.tokenizer.encode("\n\n", add_special_tokens=False)[0],
                                self.tokenizer.encode(".\n\n", add_special_tokens=False)[0],
                                self.tokenizer.encode("?\n\n", add_special_tokens=False)[0]
                            )
                            line_break_mask = (
                                batch.batch["responses"] == line_break_id[0]
                            ) | (
                                batch.batch["responses"] == line_break_id[1]
                            ) | (
                                batch.batch["responses"] == line_break_id[2]
                            )
                            line_break_mask = torch.roll(line_break_mask, 1, 1)
                            line_break_mask[:, 0] = 0
                            entropy_mask = think_mask & line_break_mask.bool()

                            forking_token_indices = torch.topk(
                                entropys * entropy_mask, 
                                self.config.algorithm.num_forking_token, 
                                dim=1
                            ).indices
                            forking_token_indices = torch.where(
                                entropy_mask.gather(1, forking_token_indices),
                                forking_token_indices,
                                0,
                            ).sort(dim=1).values

                            forking_token_indices = forking_token_indices.clone()
                            forking_token_indices = F.pad(forking_token_indices, (1, 0), value=0)
                            forking_token_indices = torch.cat(
                                [
                                    forking_token_indices, 
                                    (think_mask.sum(1, keepdim=True) + 1) \
                                        .clamp_max(valid_response_len.unsqueeze(1))
                                ], dim=1
                            )
                            
                            batch_idxs = torch.arange(forking_token_indices.size(0)).to(forking_token_indices.device)
                            batch_idxs = einops.repeat(batch_idxs, "b -> b n", n=forking_token_indices.size(1))

                            forking_token_indices = torch.stack(
                                [batch_idxs, forking_token_indices], 
                                dim=2
                            ).view(-1, 2).unique(dim=0)

                            aux_batch = deepcopy(batch)
                            aux_batch = aux_batch.sample_level_repeat(forking_token_indices[:, 0].bincount())

                            ground_truth_template = "\\boxed{{{}}}"
                            prob_tokens = [
                                torch.tensor(self.tokenizer.encode(
                                    ground_truth_template.format(item["ground_truth"])
                                )).to(entropys.device)
                                for item in aux_batch.non_tensor_batch["reward_model"]
                            ]
                            prob_token_lens = torch.tensor(list(map(len, prob_tokens))).to(entropys.device)

                            max_prob_token_len = prob_token_lens.max()
                            indices = torch.arange(max_prob_token_len).unsqueeze(0).to(entropys.device)
                            pad_token_id = self.tokenizer.pad_token_id
                            aux_batch.batch["responses"] = F.pad(
                                aux_batch.batch["responses"], 
                                (0, max_prob_token_len), 
                                value=pad_token_id
                            )
                            aux_batch.batch["input_ids"] = F.pad(
                                aux_batch.batch["input_ids"], 
                                (0, max_prob_token_len),  
                                value=pad_token_id
                            )
                            aux_batch.batch["attention_mask"] = F.pad(
                                aux_batch.batch["attention_mask"], 
                                (0, max_prob_token_len), 
                                value=0
                            )
                            aux_batch.batch["position_ids"] = torch.cat(
                                [
                                    aux_batch.batch["position_ids"], 
                                    indices + aux_batch.batch["position_ids"][:, -1:] + 1
                                ], 
                                dim=1
                            )
                            
                            seq_indices = torch.arange(seq_len + max_prob_token_len).to(entropys.device)
                            seq_indices = einops.repeat(seq_indices, "s -> b s", b=len(forking_token_indices))
 
                            cols = [torch.arange(l).to(entropys.device) for l in prob_token_lens]
                            cols = torch.cat(cols, dim=0)
                            cols = cols + forking_token_indices[:, 1].repeat_interleave(prob_token_lens) + prompt_len
                            rows = torch.arange(len(forking_token_indices)).to(entropys.device)
                            rows = rows.repeat_interleave(prob_token_lens)
                            aux_batch.batch["input_ids"].index_put_(
                                [rows, cols],
                                torch.cat(prob_tokens, dim=0),
                            )
                            
                            l = forking_token_indices[:, 1] + prompt_len
                            r = forking_token_indices[:, 1] + prob_token_lens + prompt_len
                            p_mask = seq_indices >= r.unsqueeze(1)
                            v_mask = (seq_indices >= prompt_len) & (seq_indices < r.unsqueeze(1))
                            aux_batch.batch["input_ids"][p_mask] = self.tokenizer.pad_token_id
                            aux_batch.batch["attention_mask"] = aux_batch.batch["input_ids"] != self.tokenizer.pad_token_id

                            aux_log_prob = self.actor_rollout_wg.compute_log_prob(aux_batch)

                            repro_value_mask = (seq_indices >= l.unsqueeze(1)) & (seq_indices < r.unsqueeze(1))
                            repro_value_mask = repro_value_mask[:, prompt_len:]
                            repro_value_mask_sum = repro_value_mask.sum(1)
                            assert torch.all(repro_value_mask_sum == prob_token_lens)
                            repro_value = -(aux_log_prob.batch["old_log_probs"] * repro_value_mask).sum(1) / (repro_value_mask_sum + 1e-6)
                            repro_value = repro_value.detach()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if use_co_grpo:
                                # Map experimental stream to standard keys for ref policy
                                ref_batch = batch.select(
                                    batch_keys=["exp_input_ids", "exp_attention_mask", "exp_position_ids"],
                                    non_tensor_batch_keys=["uid"]
                                )
                                ref_batch.batch["input_ids"] = ref_batch.batch.pop("exp_input_ids")
                                ref_batch.batch["attention_mask"] = ref_batch.batch.pop("exp_attention_mask")
                                ref_batch.batch["position_ids"] = ref_batch.batch.pop("exp_position_ids")
                                ref_batch.batch["responses"] = batch.batch["exp_responses"]
                                target_batch = ref_batch
                            else:
                                target_batch = batch

                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(target_batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(target_batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        if use_co_grpo:
                            # Already computed exp_token_rewards above
                            reward_extra_infos_dict: dict[str, list] = reward_extra_infos_dict if 'reward_extra_infos_dict' in locals() else {}
                        else:
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.adv_estimator in [
                            AdvantageEstimator.REPRO_PPO,
                            AdvantageEstimator.REPRO_GRPO,
                            AdvantageEstimator.REPRO_REINFORCE_PLUS_PLUS,
                            AdvantageEstimator.REPRO_REINFORCE_PLUS_PLUS_BASELINE
                        ]:
                            def kendall_batch(x_batch, y_batch):
                                """
                                Calculates Kendall's Tau correlation coefficient for a batch of (x, y) pairs.

                                Args:
                                    x_batch (torch.Tensor): A batch of x-sequences, shape (batch_size, sequence_length).
                                    y_batch (torch.Tensor): A batch of y-sequences, shape (batch_size, sequence_length).

                                Returns:
                                    torch.Tensor: A 1D tensor of Kendall's Tau values, shape (batch_size,).
                                                Returns NaN for sequences with less than 2 elements.
                                """
                                batch_size, n = x_batch.shape

                                # Reshape x_batch and y_batch for pairwise comparisons
                                # x_batch_expanded_i: (batch_size, n, 1) - for x_i
                                # x_batch_expanded_j: (batch_size, 1, n) - for x_j
                                x_batch_expanded_i = x_batch.unsqueeze(2) # Adds a dimension at position 2
                                x_batch_expanded_j = x_batch.unsqueeze(1) # Adds a dimension at position 1

                                y_batch_expanded_i = y_batch.unsqueeze(2)
                                y_batch_expanded_j = y_batch.unsqueeze(1)

                                # Calculate pairwise differences and their signs
                                # Shape of diff_x and diff_y: (batch_size, n, n)
                                diff_x = (x_batch_expanded_i - x_batch_expanded_j).sign()
                                diff_y = (y_batch_expanded_i - y_batch_expanded_j).sign()

                                # Element-wise product of signs
                                # This gives 1 for concordant, -1 for discordant, 0 for ties
                                # Shape: (batch_size, n, n)
                                concordance_matrix = diff_x * diff_y

                                # Sum across the n x n matrix for each batch item
                                # Shape: (batch_size,)
                                numerator = concordance_matrix.sum(dim=(1, 2))

                                # Total number of unique pairs (excluding self-pairs i=j) for each sequence
                                # Shape: (batch_size,)
                                denominator = n * (n - 1)

                                # Calculate Kendall's Tau
                                # Shape: (batch_size,)
                                tau_values = numerator / denominator

                                return tau_values
                            
                            groups = forking_token_indices[:, 0]
                            values = forking_token_indices[:, 1]
                            unique_groups = torch.unique(groups)
                            repro_scores = []

                            for g in unique_groups:
                                group_values = repro_value[groups == g]
                                repro_scores.append(group_values)

                            max_len = max(len(x) for x in repro_scores)
                            repro_scores = torch.stack([torch.cat([x, x.new_zeros(max_len - len(x))]) for x in repro_scores])
                            repro_scores_mask = repro_scores != 0

                            def perform_ema(values: torch.Tensor, alpha: float = 0.01):
                                ema_values = torch.zeros_like(values)
                                ema_values[:, 0] = values[:, 0]
                                for i in range(1, values.size(1)):
                                    ema_values[:, i] = alpha * values[:, i] + (1 - alpha) * ema_values[:, i - 1]
                                return ema_values
                            
                            ema_repro_scores = perform_ema(repro_scores)

                            repro_scores = (repro_scores[:, 0: 1] - repro_scores[:, 1:]) / (repro_scores[:, 0: 1] + 1e-6)
                            repro_scores = F.tanh(repro_scores - 1) + 1 

                            mono_scores = []
                            for i in range(1, ema_repro_scores.size(1)):
                                x = torch.arange(1, i + 2).unsqueeze(0).repeat(len(ema_repro_scores), 1).to(ema_repro_scores.device)
                                mono_scores.append(-kendall_batch(x, ema_repro_scores[:, :i + 1]) + 1 / 2)
                            mono_scores = torch.stack(mono_scores, dim=1)
                            repro_scores = repro_scores # + 0.1 * mono_scores

                            repro_scores = F.pad(repro_scores, (1, 0)).diff(1)
                            repro_scores = F.pad(repro_scores, (1, 0))
                            
                            batch.batch["token_level_repro_rewards"] = torch.zeros_like(batch.batch["token_level_scores"])
                            batch.batch["token_level_repro_rewards"][groups, values - 1] = repro_scores[repro_scores_mask]

                            batch.batch["repro_mask"] = torch.zeros_like(batch.batch["token_level_scores"])
                            batch.batch["repro_mask"][groups, values - 1] = 1
                            batch.batch["repro_mask"][:, 0] = 0
                            # batch.batch["repro__mask"] = batch.batch["repro_mask"] * (batch.batch["token_level_scores"].sum(-1, keepdim=True) > 0)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        # Calculate training progress for curriculum learning
                        training_progress = min(1.0, self.global_steps / self.total_training_steps)
                        
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                            training_progress=training_progress,
                        )
                        
                        # Log Co-GRPO specific metrics
                        if use_co_grpo and 'co_grpo_metadata' in batch.meta_info:
                            co_grpo_meta = batch.meta_info['co_grpo_metadata']
                            metrics['co_grpo/training_progress'] = co_grpo_meta['training_progress']
                            metrics['co_grpo/effective_control_weight'] = co_grpo_meta['effective_control_weight']
                            metrics['co_grpo/effective_exp_weight'] = 1.0 - co_grpo_meta['effective_control_weight']
                            
                            # Log advantage statistics for both streams
                            if 'advantages' in batch.batch:
                                policy_advantages = batch.batch['advantages']
                                metrics['co_grpo/policy_advantage_mean'] = policy_advantages.mean().item()
                                metrics['co_grpo/policy_advantage_std'] = policy_advantages.std().item()
                            
                            if 'verifier_advantages' in batch.batch:
                                verifier_advantages = batch.batch['verifier_advantages']
                                metrics['co_grpo/verifier_advantage_mean'] = verifier_advantages.mean().item()
                                metrics['co_grpo/verifier_advantage_std'] = verifier_advantages.std().item()

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            if use_co_grpo:
                                # Map experimental stream to standard keys for actor update
                                batch.batch["responses"] = batch.batch["exp_responses"]
                                batch.batch["attention_mask"] = batch.batch["exp_attention_mask"]
                                batch.batch["input_ids"] = batch.batch["exp_input_ids"]
                                batch.batch["position_ids"] = batch.batch["exp_position_ids"]
                                # Ensure prompts exist for logging / decoding
                                if "exp_prompts" in batch.batch:
                                    batch.batch["prompts"] = batch.batch["exp_prompts"]
                                else:
                                    exp_prompt_len = batch.batch["exp_input_ids"].size(1) - batch.batch["exp_responses"].size(1)
                                    batch.batch["prompts"] = batch.batch["exp_input_ids"][..., :exp_prompt_len]
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    rollout_dump_freq = self.config.trainer.get("rollout_dump_freq", 0)  # 0 means every step (backward compatible)
                    # rollout_dump_answer = self.config.trainer.get("rollout_dump_answer", None)  # 0 means every step (backward compatible)
                    # Only dump if: 1) rollout_data_dir is set, AND 2) either freq is 0 or it's time to dump
                    should_dump_rollout = rollout_data_dir and (rollout_dump_freq == 0 or self.global_steps % rollout_dump_freq == 0 or is_last_step)
                    if should_dump_rollout:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            print("non_tensor_batch keys:", list(batch.non_tensor_batch.keys()) if hasattr(batch, 'non_tensor_batch') else "N/A")
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            # For Co-GRPO, also dump control/exp streams separately to compare intervention effect
                            extra_infos = dict(reward_extra_infos_dict)
                            if use_co_grpo:
                                control_prompt_len = batch.batch["control_input_ids"].size(1) - batch.batch["control_responses"].size(1)
                                control_prompts = batch.batch["control_input_ids"][..., :control_prompt_len]
                                exp_prompt_len = batch.batch["exp_input_ids"].size(1) - batch.batch["exp_responses"].size(1)
                                exp_prompts = batch.batch["exp_input_ids"][..., :exp_prompt_len]

                                extra_infos["control_prompt"] = self.tokenizer.batch_decode(control_prompts, skip_special_tokens=True)
                                extra_infos["exp_prompt"] = self.tokenizer.batch_decode(exp_prompts, skip_special_tokens=True)
                                extra_infos["control_output"] = self.tokenizer.batch_decode(batch.batch["control_responses"], skip_special_tokens=True)
                                extra_infos["exp_output"] = self.tokenizer.batch_decode(batch.batch["exp_responses"], skip_special_tokens=True)

                                # Add Verifier outputs for analysis
                                if hasattr(batch, 'non_tensor_batch'):
                                    if "critiques" in batch.non_tensor_batch:
                                        extra_infos["verifier_critiques"] = batch.non_tensor_batch["critiques"].tolist()
                                    if "hints" in batch.non_tensor_batch:
                                        extra_infos["verifier_hints"] = batch.non_tensor_batch["hints"].tolist()
                                    if "num_interventions" in batch.non_tensor_batch:
                                        extra_infos["num_interventions"] = batch.non_tensor_batch["num_interventions"].tolist()
                                    if "hint_token_counts" in batch.non_tensor_batch:
                                        extra_infos["hint_token_counts"] = batch.non_tensor_batch["hint_token_counts"].tolist()

                            # answers = batch.batch["responses"].cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=extra_infos,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        return batch


# print("split thinking...")

                            # entropy_mask = torch.arange(entropys.size(1), device=entropys.device).unsqueeze(0).repeat(entropys.size(0), 1)
                            # entropy_mask = entropy_mask < (len_response_before_think_end.unsqueeze(1) - 1)

                            # oscillation_token_indices = torch.topk(
                            #     entropys * entropy_mask, 
                            #     self.config.algorithm.num_oscillation_token, 
                            #     dim=1
                            # ).indices.sort(dim=1).values
                            # clamp_max_value = entropy_mask.sum(1) - 1
                            # oscillation_token_indices.clamp_max(
                            #     clamp_max_value.unsqueeze(1).expand_as(oscillation_token_indices))

                            # split_token_indices = oscillation_token_indices.clone()
                            # split_token_indices = F.pad(split_token_indices, (1, 0), value=0)
                            # split_token_indices = torch.cat(
                            #     [split_token_indices, entropy_mask.sum(1, keepdim=True).clamp_max(valid_response_len.unsqueeze(1) - 1)], dim=1
                            # )

                            # print("get topk entropys...")

                            # ids = batch.batch["responses"].unsqueeze(1).repeat((1, self.config.algorithm.num_oscillation_token + 2, 1)).reshape(-1, response_len)
                            # ids_indices = torch.arange(response_len, device=ids.device).unsqueeze(0).repeat((batch_size * (self.config.algorithm.num_oscillation_token + 2), 1))
                            # ids_mask = ids_indices < (split_token_indices.flatten().unsqueeze(1))
                            # ids[~ids_mask] = self.tokenizer.pad_token_id
                            
                            # answers = [item["ground_truth"] for item in batch.non_tensor_batch["reward_model"]]
                            # answers = [answer for answer in answers for _ in range(self.config.algorithm.num_oscillation_token + 2)]
                            # answer_lens = torch.tensor([len(self.tokenizer.encode(answer, add_special_tokens=False)) for answer in answers])
                            
                            # seqs = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
                            # seqs = [seq + answer for seq, answer in zip(seqs, answers)]

                            # encode_seqs = self.tokenizer(
                            #     seqs, 
                            #     add_special_tokens=False, 
                            #     padding=True, 
                            #     truncation=False, 
                            #     return_tensors="pt"
                            # ).to(entropys.device)
                            # completion_input_ids = torch.cat([batch.batch["prompts"], encode_seqs["input_ids"]], dim=1)
                            # completion_attention_mask = torch.cat([batch.batch["attention_mask"][:, :prompt_len], encode_seqs["attention_mask"]], dim=1)
                            # completion_position_ids = compute_position_id_with_mask(completion_attention_mask)

                            # print("Completing....")
                            # completion_out = self.actor_rollout_wg.compute_log_prob(
                            #         DataProto.from_dict(
                            #         tensors={
                            #             "input_ids": completion_input_ids,
                            #             "attention_mask": completion_attention_mask,
                            #             "position_ids": completion_position_ids,
                            #             "responses": encode_seqs["input_ids"],
                            #         },
                            #         meta_info={
                            #             "only_get_input_log_probs": True,
                            #             "micro_batch_size": self.config.rollout.log_prob_micro_batch_size_per_gpu,
                            #             "max_token_len": self.config.rollout.log_prob_max_token_len_per_gpu,
                            #             "temperature": 1.0,
                            #             "use_dynamic_bsz": self.config.rollout.log_prob_use_dynamic_bsz,
                            #         }
                            #     )
                            # )
                            # print("Done")

                            # completion_log_probs = completion_out.batch["old_log_probs"]

                            # answer_mask_indices = torch.arange(completion_log_probs.size(1)).unsqueeze(0).repeat(completion_log_probs.size(0), 1)
                            # answer_mask = (answer_mask_indices >= (encode_seqs["input_ids"].sum(1, keepdim=True) - answer_lens.unsqueeze(1))) & (answer_mask_indices < encode_seqs["input_ids"].sum(1, keepdim=True))
                            # sgrpo_value = -(completion_log_probs * answer_mask).sum(1) / (answer_mask.sum(1) + 1e-6)
                            # sgrpo_value = sgrpo_value.detach().reshape(batch_size, self.config.algorithm.num_oscillation_token + 2)
