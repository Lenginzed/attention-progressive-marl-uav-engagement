import numpy as np
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask, ResearchMultipleCombatTask


class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is an multi-player competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self._create_records = False
        self.init_states = None

    @property
    def share_observation_space(self):
        return self.task.share_observation_space

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'multiplecombat':
            self.task = MultipleCombatTask(self.config)
        elif taskname == 'research_multiplecombat':
            self.task = ResearchMultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat':
            self.task = HierarchicalMultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat_shoot':
            self.task = HierarchicalMultipleCombatShootTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        return self._pack(obs), self._pack(share_obs)

    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = {uid: sim.init_state.copy() for uid, sim in self._jsbsims.items()}
        randomization_cfg = getattr(self.config, "initial_state_randomization", None)
        for uid, sim in self._jsbsims.items():
            init_state = self.init_states[uid].copy()
            if isinstance(randomization_cfg, dict):
                init_state = self._randomize_initial_state(init_state, randomization_cfg)
            sim.reload(init_state)
        self._tempsims.clear()

    def _randomize_initial_state(self, init_state, cfg):
        randomized_state = init_state.copy()
        if "longitude_deg" in cfg:
            randomized_state["ic_long_gc_deg"] = randomized_state.get("ic_long_gc_deg", 120.0) + \
                self.np_random.uniform(-cfg["longitude_deg"], cfg["longitude_deg"])
        if "latitude_deg" in cfg:
            randomized_state["ic_lat_geod_deg"] = randomized_state.get("ic_lat_geod_deg", 60.0) + \
                self.np_random.uniform(-cfg["latitude_deg"], cfg["latitude_deg"])
        if "altitude_ft" in cfg:
            randomized_state["ic_h_sl_ft"] = randomized_state.get("ic_h_sl_ft", 20000.0) + \
                self.np_random.uniform(-cfg["altitude_ft"], cfg["altitude_ft"])
        if "heading_deg" in cfg:
            randomized_state["ic_psi_true_deg"] = (
                randomized_state.get("ic_psi_true_deg", 0.0) +
                self.np_random.uniform(-cfg["heading_deg"], cfg["heading_deg"])
            ) % 360.0
        if "speed_fps" in cfg:
            randomized_state["ic_u_fps"] = randomized_state.get("ic_u_fps", 800.0) + \
                self.np_random.uniform(-cfg["speed_fps"], cfg["speed_fps"])
        return randomized_state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict): the agents' actions, each key corresponds to an agent_id

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # apply actions
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        self.task.step(self)
        obs = self.get_obs()
        share_obs = self.get_state()

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        if np.all(self._pack(dones)):
            info = self.task.get_episode_summary(self, info)

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        for ego_id in self.ego_ids:
            rewards[ego_id] = [ego_reward]
        for enm_id in self.enm_ids:
            rewards[enm_id] = [enm_reward]

        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info
