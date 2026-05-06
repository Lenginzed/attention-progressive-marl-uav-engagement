import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


class MATD3Trainer:
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.args = args
        self.policy = policy
        self.device = device
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_delay = args.policy_delay
        self.total_updates = 0

    def _reshape_actor_actions(self, obs_tensor, actor):
        batch_size, num_agents, obs_dim = obs_tensor.shape
        flat_obs = obs_tensor.reshape(batch_size * num_agents, obs_dim)
        flat_actions = actor(flat_obs)
        return flat_actions.reshape(batch_size, num_agents, -1)

    def train(self, replay_buffer, batch_size):
        batch = replay_buffer.sample(batch_size)
        obs = _to_tensor(batch["obs"], self.device)
        state = _to_tensor(batch["share_obs"], self.device)
        actions = _to_tensor(batch["actions"], self.device)
        rewards = _to_tensor(batch["rewards"], self.device)
        dones = _to_tensor(batch["dones"], self.device)
        next_obs = _to_tensor(batch["next_obs"], self.device)
        next_state = _to_tensor(batch["next_share_obs"], self.device)

        with torch.no_grad():
            next_actions = self._reshape_actor_actions(next_obs, self.policy.actor_target)
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions[..., :3] = (next_actions[..., :3] + noise[..., :3]).clamp(-1.0, 1.0)
            next_actions[..., 3] = (next_actions[..., 3] + noise[..., 3]).clamp(0.4, 0.9)
            next_joint_actions = next_actions.reshape(next_actions.shape[0], -1)
            target_q1, target_q2 = self.policy.critic_target(next_state, next_joint_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.gamma * target_q

        joint_actions = actions.reshape(actions.shape[0], -1)
        current_q1, current_q2 = self.policy.critic(state, joint_actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.policy.critic_optimizer.step()

        actor_loss_value = None
        if self.total_updates % self.policy_delay == 0:
            actor_actions = self._reshape_actor_actions(obs, self.policy.actor)
            actor_joint_actions = actor_actions.reshape(actor_actions.shape[0], -1)
            actor_loss = -self.policy.critic.q1_only(state, actor_joint_actions).mean()
            self.policy.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.policy.actor_optimizer.step()
            self.soft_update(self.policy.actor, self.policy.actor_target)
            self.soft_update(self.policy.critic, self.policy.critic_target)
            actor_loss_value = float(actor_loss.item())

        self.total_updates += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value if actor_loss_value is not None else 0.0,
            "q1_mean": float(current_q1.mean().item()),
            "q2_mean": float(current_q2.mean().item()),
        }

    def soft_update(self, source, target):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
