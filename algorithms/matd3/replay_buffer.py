import numpy as np


class MATD3ReplayBuffer:
    def __init__(self, capacity, num_agents, obs_dim, share_obs_dim, action_dim):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.share_obs_dim = int(share_obs_dim)
        self.action_dim = int(action_dim)
        self.reset()

    def reset(self):
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.share_obs = np.zeros((self.capacity, self.share_obs_dim), dtype=np.float32)
        self.next_share_obs = np.zeros_like(self.share_obs)
        self.actions = np.zeros((self.capacity, self.num_agents, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def insert_batch(self, obs, share_obs, actions, rewards, dones, next_obs, next_share_obs):
        batch_size = int(obs.shape[0])
        indices = (np.arange(batch_size) + self.ptr) % self.capacity
        self.obs[indices] = obs
        self.share_obs[indices] = share_obs
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.dones[indices] = dones
        self.next_obs[indices] = next_obs
        self.next_share_obs[indices] = next_share_obs
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size, rng=None):
        generator = np.random.default_rng() if rng is None else rng
        indices = generator.integers(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[indices],
            "share_obs": self.share_obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "next_obs": self.next_obs[indices],
            "next_share_obs": self.next_share_obs[indices],
        }

