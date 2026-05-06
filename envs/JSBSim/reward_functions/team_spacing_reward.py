import numpy as np

from .reward_function_base import BaseRewardFunction


class TeamSpacingReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.target_distance = getattr(self.config, f"{self.__class__.__name__}_target_distance", 1800.0)
        self.min_distance = getattr(self.config, f"{self.__class__.__name__}_min_distance", 500.0)
        self.max_distance = getattr(self.config, f"{self.__class__.__name__}_max_distance", 6000.0)
        self.sigma = getattr(self.config, f"{self.__class__.__name__}_sigma", 1200.0)
        self.reward_item_names = [self.__class__.__name__, self.__class__.__name__ + "_spacing"]

    def get_reward(self, task, env, agent_id):
        partners = env.agents[agent_id].partners
        if len(partners) == 0:
            return self._process(0.0, agent_id, (0.0,))

        partner_distance = np.linalg.norm(partners[0].get_position() - env.agents[agent_id].get_position())
        spacing_reward = np.exp(-((partner_distance - self.target_distance) / max(self.sigma, 1.0)) ** 2)
        if partner_distance < self.min_distance:
            spacing_reward -= (self.min_distance - partner_distance) / max(self.min_distance, 1.0)
        if partner_distance > self.max_distance:
            spacing_reward -= (partner_distance - self.max_distance) / max(self.max_distance, 1.0)
        return self._process(spacing_reward, agent_id, (spacing_reward,))
