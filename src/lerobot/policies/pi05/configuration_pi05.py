#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # MEM short-horizon observation memory (https://arxiv.org/abs/2603.03596).
    # Historical image tokens are fused inside SigLIP and dropped before the
    # language backbone. Historical proprioceptive states become one continuous
    # backbone token per frame. Both paths are opt-in and independent.
    #
    # MEM pre-trains on six observations spaced one second apart. `memory_stride` is
    # counted in dataset frames, so the default matches that spacing only at 30 fps,
    # the usual LeRobot recording rate. Scale it with the dataset: a 10 fps dataset
    # such as `lerobot/robomme` needs `memory_stride=10` for the same one second.
    use_visual_memory: bool = False
    use_proprioceptive_memory: bool = False
    memory_frames: int = 6
    memory_stride: int = 30
    memory_temporal_attention_every: int = 4

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Conditional Trajectory Peaks (CTP). When enabled, Pi0.5 keeps its
    # vision-language/action-expert backbone but predicts an entire
    # trajectory mixture in one forward pass instead of running flow ODE
    # denoising. Disabled by default to preserve legacy Pi0.5 behavior.
    ctp_enabled: bool = False
    ctp_num_modes: int = 4
    ctp_min_width: float = 0.02
    ctp_max_width: float = 0.5
    ctp_initial_width: float = 0.15
    # Weight in per-coordinate NLL units (joint-NLL paper lambda / H*D).
    ctp_overlap_weight: float = 0.1
    ctp_head_init_std: float = 0.005
    ctp_mask_padding: bool = False
    # V3 is opt-in; old fused heads keep their original state-dict layout.
    ctp_split_head: bool = False
    ctp_detach_width_context: bool = False
    ctp_optimizer_logits_lr_scale: float = 1.0
    ctp_optimizer_width_lr_scale: float = 1.0
    ctp_temperature_initial: float = 1.0
    ctp_temperature_final: float = 1.0
    ctp_temperature_anneal_steps: int = 1500
    ctp_overlap_detach_probabilities: bool = False
    ctp_optimizer_vlm_lr_scale: float = 1.0
    ctp_optimizer_head_lr_scale: float = 1.0

    # Anti-collapse controls.  Legacy defaults exactly preserve existing CTP
    # checkpoints; each experiment must explicitly record its chosen controls.
    ctp_probability_floor: float = 0.0
    ctp_entropy_target_effective_modes: float = 1.0
    ctp_entropy_weight: float = 0.0
    ctp_overlap_probability_weighted: bool = True
    ctp_overlap_detach_widths: bool = False

    # Training-free, permutation-equivariant temporal decoding.  ``top1`` is
    # the legacy behavior.  ``path_belief`` transports belief according to the
    # overlap between the previous unexecuted suffix and current candidate
    # prefixes, never according to component index.
    ctp_inference_mode: str = "top1"
    ctp_belief_self_transition: float = 0.95
    ctp_belief_observation_weight: float = 1.0
    ctp_belief_release_ratio: float = 4.0
    ctp_belief_transport_temperature: float = 0.1
    ctp_belief_confidence_threshold: float = 0.0
    ctp_belief_probability_floor: float = 1.0e-4
    ctp_belief_cumulative_action_dims: int = 6

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None
    # Maximum clean action-prefix length sampled during training. Zero disables trained RTC.
    rtc_training_max_delay: int = 0

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`
    text_tokenizer_name: str = "google/paligemma-3b-pt-224"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if not 0 <= self.rtc_training_max_delay < self.chunk_size:
            raise ValueError(
                "rtc_training_max_delay must satisfy "
                f"0 <= delay < chunk_size ({self.chunk_size}), got {self.rtc_training_max_delay}"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.memory_frames < 1:
            raise ValueError("memory_frames must be at least 1")
        if self.memory_stride < 1:
            raise ValueError("memory_stride must be at least 1")
        if self.memory_temporal_attention_every < 1:
            raise ValueError("memory_temporal_attention_every must be at least 1")

        if self.ctp_enabled:
            if self.ctp_detach_width_context and not self.ctp_split_head:
                raise ValueError("Detached width context requires the split head")
            if self.ctp_temperature_anneal_steps <= 0:
                raise ValueError("CTP temperature anneal steps must be positive")
            for name in (
                "ctp_head_init_std",
                "ctp_optimizer_vlm_lr_scale",
                "ctp_optimizer_head_lr_scale",
                "ctp_optimizer_logits_lr_scale",
                "ctp_optimizer_width_lr_scale",
                "ctp_temperature_initial",
                "ctp_temperature_final",
            ):
                value = getattr(self, name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
            if self.ctp_num_modes < 1:
                raise ValueError("ctp_num_modes must be positive.")
            if not 0.0 < self.ctp_min_width < self.ctp_initial_width < self.ctp_max_width:
                raise ValueError("CTP widths must satisfy 0 < min < initial < max.")
            if not math.isfinite(self.ctp_overlap_weight) or self.ctp_overlap_weight < 0.0:
                raise ValueError("ctp_overlap_weight must be finite and non-negative.")
            if not 0.0 <= self.ctp_probability_floor < 1.0:
                raise ValueError("ctp_probability_floor must be in [0, 1).")
            if not 1.0 <= self.ctp_entropy_target_effective_modes <= float(self.ctp_num_modes):
                raise ValueError("ctp_entropy_target_effective_modes must be in [1, ctp_num_modes].")
            if not math.isfinite(self.ctp_entropy_weight) or self.ctp_entropy_weight < 0.0:
                raise ValueError("ctp_entropy_weight must be finite and non-negative.")
            if self.ctp_inference_mode not in ("top1", "path_belief"):
                raise ValueError("ctp_inference_mode must be 'top1' or 'path_belief'.")
            if not 0.0 <= self.ctp_belief_self_transition <= 1.0:
                raise ValueError("ctp_belief_self_transition must be in [0, 1].")
            if (
                not math.isfinite(self.ctp_belief_observation_weight)
                or self.ctp_belief_observation_weight <= 0.0
            ):
                raise ValueError("ctp_belief_observation_weight must be finite and positive.")
            if self.ctp_belief_release_ratio != 0.0 and (
                not math.isfinite(self.ctp_belief_release_ratio) or self.ctp_belief_release_ratio < 1.0
            ):
                raise ValueError("ctp_belief_release_ratio must be 0 or at least 1.")
            if (
                not math.isfinite(self.ctp_belief_transport_temperature)
                or self.ctp_belief_transport_temperature <= 0.0
            ):
                raise ValueError("ctp_belief_transport_temperature must be finite and positive.")
            if not 0.0 <= self.ctp_belief_confidence_threshold <= 1.0:
                raise ValueError("ctp_belief_confidence_threshold must be in [0, 1].")
            if not 0.0 <= self.ctp_belief_probability_floor < 1.0:
                raise ValueError("ctp_belief_probability_floor must be in [0, 1).")
            action_feature = self.output_features.get(ACTION)
            action_dim = int(action_feature.shape[0]) if action_feature is not None else self.max_action_dim
            if not 0 <= self.ctp_belief_cumulative_action_dims <= action_dim:
                raise ValueError("ctp_belief_cumulative_action_dims must be in [0, action_dim].")
            if self.ctp_enabled and (self.use_visual_memory or self.use_proprioceptive_memory):
                raise ValueError("CTP port currently requires single-observation Pi0.5.")
            if self.ctp_enabled and (
                self.rtc_training_max_delay > 0 or (self.rtc_config is not None and self.rtc_config.enabled)
            ):
                raise ValueError("CTP replaces flow matching and is incompatible with RTC.")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def image_observation_delta_indices(self) -> list[int] | None:
        if not self.use_visual_memory:
            return None
        horizon = (self.memory_frames - 1) * self.memory_stride
        return list(range(-horizon, 1, self.memory_stride))

    @property
    def state_observation_delta_indices(self) -> list[int] | None:
        if not self.use_proprioceptive_memory:
            return None
        horizon = (self.memory_frames - 1) * self.memory_stride
        return list(range(-horizon, 1, self.memory_stride))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
