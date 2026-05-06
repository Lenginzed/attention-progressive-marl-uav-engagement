import copy
import numpy as np
import torch

from .networks import MATD3Actor, MATD3Critic, joint_action_dim_from_agents
from ..utils.flatten import build_flattener


class MATD3Policy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.cent_obs_space = cent_obs_space
        self.act_space = act_space
        self.action_dim = 4
        self.obs_dim = build_flattener(obs_space).size
        self.share_obs_dim = build_flattener(cent_obs_space).size

        self.actor = MATD3Actor(args, obs_space, self.action_dim, device=device)
        self.actor_target = copy.deepcopy(self.actor)
        joint_action_dim = joint_action_dim_from_agents(num_agents, self.action_dim)
        self.critic = MATD3Critic(args, cent_obs_space, joint_action_dim, device=device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    def act(self, obs, deterministic=True, noise_std=0.0):
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(obs).detach().cpu().numpy()
        if not deterministic and noise_std > 0.0:
            noise = np.random.normal(0.0, noise_std, size=actions.shape).astype(np.float32)
            actions[..., :3] = np.clip(actions[..., :3] + noise[..., :3], -1.0, 1.0)
            actions[..., 3] = np.clip(actions[..., 3] + noise[..., 3], 0.4, 0.9)
        return actions

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

