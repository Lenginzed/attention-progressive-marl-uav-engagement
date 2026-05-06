import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from ..utils.attention import MultiDimensionalAttentionBase
from ..utils.flatten import build_flattener
from ..utils.mlp import MLPBase, MLPLayer


def _space_dim(space):
    flattener = build_flattener(space)
    return flattener.size


class MATD3Actor(nn.Module):
    def __init__(self, args, obs_space, act_dim, device=torch.device("cpu")):
        super().__init__()
        self.use_attention_policy = args.use_attention_policy
        self.tpdv = dict(dtype=torch.float32, device=device)
        if self.use_attention_policy:
            self.base = MultiDimensionalAttentionBase(
                obs_space,
                args.hidden_size,
                args.activation_id,
                args.use_feature_normalization,
                embed_dim=args.attention_embed_dim,
                num_heads=args.attention_num_heads,
                dropout=args.attention_dropout,
                sparse_topk=args.attention_sparse_topk,
                query_token_count=1,
            )
        else:
            self.base = MLPBase(obs_space, args.hidden_size, args.activation_id, args.use_feature_normalization)
        self.head = nn.Sequential(
            nn.Linear(self.base.output_size, int(args.act_hidden_size.split(" ")[-1])),
            [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][args.activation_id],
            nn.Linear(int(args.act_hidden_size.split(" ")[-1]), act_dim),
            nn.Tanh(),
        )
        self.to(device)

    def forward(self, obs):
        obs = torch.as_tensor(obs, **self.tpdv)
        features = self.base(obs)
        raw = self.head(features)
        action = raw.clone()
        action[..., 3] = 0.65 + 0.25 * raw[..., 3]
        return action


class MATD3Critic(nn.Module):
    def __init__(self, args, state_space, joint_action_dim, device=torch.device("cpu")):
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        state_dim = _space_dim(state_space)
        self.q1 = MLPLayer(state_dim + joint_action_dim, args.hidden_size, args.activation_id)
        self.q1_out = nn.Linear(self.q1.output_size, 1)
        self.q2 = MLPLayer(state_dim + joint_action_dim, args.hidden_size, args.activation_id)
        self.q2_out = nn.Linear(self.q2.output_size, 1)
        self.to(device)

    def forward(self, state, joint_action):
        state = torch.as_tensor(state, **self.tpdv)
        joint_action = torch.as_tensor(joint_action, **self.tpdv)
        critic_input = torch.cat([state, joint_action], dim=-1)
        q1 = self.q1_out(self.q1(critic_input))
        q2 = self.q2_out(self.q2(critic_input))
        return q1, q2

    def q1_only(self, state, joint_action):
        state = torch.as_tensor(state, **self.tpdv)
        joint_action = torch.as_tensor(joint_action, **self.tpdv)
        critic_input = torch.cat([state, joint_action], dim=-1)
        return self.q1_out(self.q1(critic_input))


def joint_action_dim_from_agents(num_agents, action_dim):
    return int(num_agents * action_dim)

