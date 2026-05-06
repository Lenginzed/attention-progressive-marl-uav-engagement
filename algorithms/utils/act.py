import gymnasium as gym
import torch
import torch.nn as nn

from .distributions import BetaShootBernoulli, Categorical, DiagGaussian, Bernoulli
from .mlp import MLPLayer


def _parse_progressive_strides(config_text, action_dims):
    if config_text is None:
        return []
    stages = [stage.strip() for stage in str(config_text).split("|") if stage.strip()]
    parsed = []
    for stage in stages:
        stage_values = [max(int(value), 1) for value in stage.split()]
        if len(stage_values) == 1 and len(action_dims) > 1:
            stage_values = stage_values * len(action_dims)
        if len(stage_values) != len(action_dims):
            raise ValueError(
                f"Progressive discretization expects {len(action_dims)} stride values per stage, got {stage_values}."
            )
        parsed.append(stage_values)
    return parsed


def _parse_stage_boundaries(boundary_text, num_stages):
    if num_stages <= 0:
        return []
    if boundary_text is None:
        return [idx / num_stages for idx in range(num_stages)]
    boundaries = [float(value) for value in str(boundary_text).split() if value]
    if len(boundaries) != num_stages:
        raise ValueError(
            f"Progressive discretization expects {num_stages} stage boundaries, got {boundaries}."
        )
    return boundaries


class ACTLayer(nn.Module):
    def __init__(self, act_space, input_dim, hidden_size, activation_id, gain, progressive_cfg=None):
        super(ACTLayer, self).__init__()
        self._mlp_actlayer = False
        self._continuous_action = False
        self._multidiscrete_action = False
        self._mixed_action = False
        self._shoot_action = False
        self._use_progressive_action_discretization = False
        self._progressive_action_dims = []
        self._progressive_action_strides = []
        self._progressive_stage_boundaries = []
        self._current_progressive_stage = 0

        if len(hidden_size) > 0:
            self._mlp_actlayer = True
            self.mlp = MLPLayer(input_dim, hidden_size, activation_id)
            input_dim = self.mlp.output_size

        if isinstance(act_space, gym.spaces.Discrete):
            action_dim = act_space.n
            self.action_out = Categorical(input_dim, action_dim, gain)
        elif isinstance(act_space, gym.spaces.Box):
            self._continuous_action = True
            action_dim = act_space.shape[0]
            self.action_out = DiagGaussian(input_dim, action_dim, gain)
        elif isinstance(act_space, gym.spaces.MultiBinary):
            action_dim = act_space.shape[0]
            self.action_out = Bernoulli(input_dim, action_dim, gain)
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            self._multidiscrete_action = True
            action_dims = act_space.nvec
            action_outs = []
            for action_dim in action_dims:
                action_outs.append(Categorical(input_dim, action_dim, gain))
            self.action_outs = nn.ModuleList(action_outs)
            self._configure_progressive_discretization(progressive_cfg, action_dims)
        elif isinstance(act_space, gym.spaces.Tuple) and  \
              isinstance(act_space[0], gym.spaces.MultiDiscrete) and \
                  isinstance(act_space[1], gym.spaces.Discrete):
            # NOTE: only for shoot missile
            self._shoot_action = True
            discrete_dims = act_space[0].nvec
            self._discrete_dim = act_space[0].shape[0]
            self._control_shoot_dim = 2
            self._shoot_dim = 1
            action_outs = []
            for discrete_dim in discrete_dims:
                action_outs.append(Categorical(input_dim, discrete_dim, gain))
            action_outs.append(BetaShootBernoulli(input_dim, self._control_shoot_dim, gain))
            self.action_outs = nn.ModuleList(action_outs)
            self._configure_progressive_discretization(progressive_cfg, discrete_dims)
        else: 
            raise NotImplementedError(f"Unsupported action space type: {type(act_space)}!")

    def _configure_progressive_discretization(self, progressive_cfg, action_dims):
        if progressive_cfg is None or not progressive_cfg.get("enabled", False):
            return
        self._progressive_action_dims = list(map(int, action_dims))
        self._progressive_action_strides = _parse_progressive_strides(
            progressive_cfg.get("strides"), self._progressive_action_dims
        )
        if len(self._progressive_action_strides) == 0:
            return
        self._progressive_stage_boundaries = _parse_stage_boundaries(
            progressive_cfg.get("boundaries"), len(self._progressive_action_strides)
        )
        self._use_progressive_action_discretization = True
        self._current_progressive_stage = 0

    def set_progress(self, progress: float):
        if not self._use_progressive_action_discretization:
            return
        progress = min(max(float(progress), 0.0), 1.0)
        stage = 0
        for idx, boundary in enumerate(self._progressive_stage_boundaries):
            if progress >= boundary:
                stage = idx
        self._current_progressive_stage = stage

    def _build_action_masks(self, batch_size, device):
        if not self._use_progressive_action_discretization:
            return None
        stage_strides = self._progressive_action_strides[self._current_progressive_stage]
        action_masks = []
        for action_dim, stride in zip(self._progressive_action_dims, stage_strides):
            mask = torch.zeros((batch_size, action_dim), dtype=torch.bool, device=device)
            allowed_indices = list(range(0, action_dim, max(int(stride), 1)))
            if allowed_indices[-1] != action_dim - 1:
                allowed_indices.append(action_dim - 1)
            mask[:, allowed_indices] = True
            action_masks.append(mask)
        return action_masks

    def forward(self, x, deterministic=False, **kwargs):
        """
        Compute actions and action logprobs from given input.

        Args:
            x (torch.Tensor): input to network.
            deterministic (bool): whether to sample from action distribution or return the mode.

        Returns:
            actions (torch.Tensor): actions to take.
            action_log_probs (torch.Tensor): log probabilities of taken actions.
        """
        if self._mlp_actlayer:
            x = self.mlp(x)

        if self._multidiscrete_action:
            actions = []
            action_log_probs = []
            action_masks = self._build_action_masks(x.size(0), x.device)
            for action_idx, action_out in enumerate(self.action_outs):
                action_mask = None if action_masks is None else action_masks[action_idx]
                action_dist = action_out(x, action_mask=action_mask)
                action = action_dist.mode() if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
        
        elif self._shoot_action:
            actions = []
            action_log_probs = []
            action_masks = self._build_action_masks(x.size(0), x.device)
            for action_idx, action_out in enumerate(self.action_outs[:-1]):
                action_mask = None if action_masks is None else action_masks[action_idx]
                action_dist = action_out(x, action_mask=action_mask)
                action = action_dist.mode() if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            shoot_action_dist = self.action_outs[-1](x, **kwargs)
            shoot_action = shoot_action_dist.mode() if deterministic else shoot_action_dist.sample()
            actions.append(shoot_action)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)

        else:
            action_dists = self.action_out(x)
            actions = action_dists.mode() if deterministic else action_dists.sample()
            action_log_probs = action_dists.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(self, x, action, active_masks=None, **kwargs):
        """
        Compute log probability and entropy of given actions.

        Args:
            x (torch.Tensor): input to network.
            action (torch.Tensor): actions whose entropy and log probability to evaluate.
            active_masks (torch.Tensor): denotes whether an agent is active or dead.

        Returns:
            action_log_probs (torch.Tensor): log probabilities of the input actions.
            dist_entropy (torch.Tensor): action distribution entropy for the given inputs.
        """
        if self._mlp_actlayer:
            x = self.mlp(x)

        if self._multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            action_masks = self._build_action_masks(x.size(0), x.device)
            for action_idx, (action_out, act) in enumerate(zip(self.action_outs, action)):
                action_mask = None if action_masks is None else action_masks[action_idx]
                action_dist = action_out(x, action_mask=action_mask)
                action_log_probs.append(action_dist.log_probs(act.unsqueeze(-1)))
                if active_masks is not None:
                    dist_entropy.append((action_dist.entropy() * active_masks) / active_masks.sum())
                else:
                    dist_entropy.append(action_dist.entropy() / action_log_probs[-1].size(0))
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            dist_entropy = torch.cat(dist_entropy, dim=-1).sum(dim=-1, keepdim=True)

        elif self._shoot_action:
            dis_action, shoot_action = action.split((self._discrete_dim, self._shoot_dim), dim=-1)
            action_log_probs = []
            dist_entropy = []
            # multi-discrete action
            dis_action = torch.transpose(dis_action, 0, 1)
            action_masks = self._build_action_masks(x.size(0), x.device)
            for action_idx, (action_out, act) in enumerate(zip(self.action_outs[:-1], dis_action)):
                action_mask = None if action_masks is None else action_masks[action_idx]
                action_dist = action_out(x, action_mask=action_mask)
                action_log_probs.append(action_dist.log_probs(act.unsqueeze(-1)))
                if active_masks is not None:
                    dist_entropy.append((action_dist.entropy() * active_masks) / active_masks.sum())
                else:
                    dist_entropy.append(action_dist.entropy() / action_log_probs[-1].size(0))

            # shoot action
            shoot_action_dist = self.action_outs[-1](x, **kwargs)
            action_log_probs.append(shoot_action_dist.log_probs(shoot_action))
            if active_masks is not None:
                dist_entropy.append((shoot_action_dist.entropy() * active_masks) / active_masks.sum())
            else:
                dist_entropy.append(shoot_action_dist.entropy() / action_log_probs[-1].size(0))

            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            dist_entropy = torch.cat(dist_entropy, dim=-1).sum(dim=-1, keepdim=True)

        else:
            action_dist = self.action_out(x)
            action_log_probs = action_dist.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_dist.entropy() * active_masks) / active_masks.sum()
            else:
                dist_entropy = action_dist.entropy() / action_log_probs.size(0)
        return action_log_probs, dist_entropy

    def get_probs(self, x):
        """
        Compute action probabilities from inputs.

        Args:
            x (torch.Tensor): input to network.

        Return:
            action_probs (torch.Tensor):
        """
        if self._mlp_actlayer:
            x = self.mlp(x)
        if self._multidiscrete_action:
            action_probs = []
            for action_out in self.action_outs:
                action_dist = action_out(x)
                action_prob = action_dist.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, dim=-1)
        elif self._continuous_action or self._shoot_action:
            raise ValueError("Normal distribution has no `probs` attribute!")
        else:
            action_dists = self.action_out(x)
            action_probs = action_dists.probs
        return action_probs

    @property
    def output_size(self) -> int:
        if self._multidiscrete_action or self._shoot_action:
            return len(self.action_outs)
        else:
            return self.action_out.output_size

    @property
    def current_progressive_stage(self) -> int:
        return self._current_progressive_stage
