import numpy as np
from gymnasium import spaces
from typing import Tuple
import torch

from .singlecombat_task import SingleCombatTask
from ..core.catalog import Catalog as c
from ..core.simulatior import MissileSimulator
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward, MissilePostureReward, RelativeAltitudeReward, TeamSpacingReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn, OutOfBounds
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor
from ..model.rule_based_opponents import build_handcrafted_opponent


class MultipleCombatTask(SingleCombatTask):
    def __init__(self, config):
        self.total_aircraft = 4
        self.use_handcrafted_opponent = getattr(config, "use_handcrafted_opponent", False)
        self.handcrafted_opponent_level = getattr(config, "handcrafted_opponent_level", "easy")
        self.handcrafted_opponent = build_handcrafted_opponent(self.handcrafted_opponent_level) \
            if self.use_handcrafted_opponent else None
        super().__init__(config)

        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            EventDrivenReward(self.config)
        ]

        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self) -> int:
        return 2 if self.use_handcrafted_opponent else self.total_aircraft

    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.obs_length = 9 + (self.total_aircraft - 1) * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.total_aircraft * self.obs_length,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        # (1) ego info normalization
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)(unit: 5G)
        # (2) relative inof w.r.t partner+enemies state
        offset = 8
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset+1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset+2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset+3] = AO
            norm_obs[offset+4] = TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        if self.use_handcrafted_opponent and agent_id in env.enm_ids:
            return self._apply_low_altitude_safety_shield(
                env,
                agent_id,
                np.asarray(self.handcrafted_opponent.get_action(env, agent_id), dtype=np.float32),
            )
        action = np.asarray(action, dtype=np.float32)
        if self._looks_like_continuous_action(action):
            norm_act = action.copy()
            norm_act[0] = np.clip(norm_act[0], -1.0, 1.0)
            norm_act[1] = np.clip(norm_act[1], -1.0, 1.0)
            norm_act[2] = np.clip(norm_act[2], -1.0, 1.0)
            norm_act[3] = np.clip(norm_act[3], 0.4, 0.9)
            return self._apply_low_altitude_safety_shield(env, agent_id, norm_act)
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return self._apply_low_altitude_safety_shield(env, agent_id, norm_act)

    def reset(self, env):
        if self.use_handcrafted_opponent:
            self.handcrafted_opponent.reset(env)
        return super().reset(env)

    def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
        if env.agents[agent_id].is_alive:
            return super().get_reward(env, agent_id, info=info)
        else:
            return 0.0, info

    def _compute_team_score(self, env, team_ids):
        target_dist = getattr(self.config, "PostureReward_target_dist", 3.0)
        scores = []
        for agent_id in team_ids:
            agent = env.agents[agent_id]
            if not agent.is_alive:
                continue
            ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
            pair_scores = []
            for enemy in agent.enemies:
                if not enemy.is_alive:
                    continue
                enm_feature = np.hstack([enemy.get_position(), enemy.get_velocity()])
                AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
                delta_altitude = np.abs(enemy.get_position()[-1] - agent.get_position()[-1])
                angle_term = 0.6 * (1.0 - AO / np.pi) + 0.4 * (TA / np.pi)
                range_term = np.exp(-((R / 1000.0 - target_dist) / max(target_dist, 1e-6)) ** 2)
                altitude_term = np.clip(1.0 - delta_altitude / 2000.0, -1.0, 1.0)
                pair_scores.append(angle_term + 0.5 * range_term + 0.3 * altitude_term)
            if pair_scores:
                scores.append(max(pair_scores))
        if len(scores) == 0:
            return 0.0
        return float(np.mean(scores))

    def get_episode_summary(self, env, info):
        agent_termination = info.get("agent_termination", {})
        ego_alive = sum(int(env.agents[agent_id].is_alive) for agent_id in env.ego_ids)
        enemy_alive = sum(int(env.agents[agent_id].is_alive) for agent_id in env.enm_ids)
        ego_score = self._compute_team_score(env, env.ego_ids) + 0.5 * ego_alive
        enemy_score = self._compute_team_score(env, env.enm_ids) + 0.5 * enemy_alive
        margin = getattr(self.config, "episode_result_margin", 0.05)

        if enemy_alive == 0 and ego_alive > 0:
            result = "win"
        elif ego_alive == 0 and enemy_alive > 0:
            result = "loss"
        elif ego_score - enemy_score > margin:
            result = "win"
        elif enemy_score - ego_score > margin:
            result = "loss"
        else:
            result = "draw"

        info["episode_summary"] = {
            "result": result,
            "steps": env.current_step,
            "ego_alive": ego_alive,
            "enemy_alive": enemy_alive,
            "ego_score": ego_score,
            "enemy_score": enemy_score,
            "ego_safety_shield_rate": self.get_safety_shield_rate(env.ego_ids),
            "enemy_safety_shield_rate": self.get_safety_shield_rate(env.enm_ids),
            "ego_end_reasons": sorted({
                reason for agent_id in env.ego_ids for reason in agent_termination.get(agent_id, [])
            }),
            "enemy_end_reasons": sorted({
                reason for agent_id in env.enm_ids for reason in agent_termination.get(agent_id, [])
            }),
        }
        return info


class ResearchMultipleCombatTask(MultipleCombatTask):
    def __init__(self, config):
        super().__init__(config)
        self.use_artillery = getattr(self.config, "use_artillery", True)
        self.timeout_win_reward = getattr(self.config, "timeout_win_reward", 120.0)
        self.timeout_loss_penalty = getattr(self.config, "timeout_loss_penalty", -120.0)
        self.timeout_draw_penalty = getattr(self.config, "timeout_draw_penalty", -20.0)
        self.artillery_kill_reward = getattr(self.config, "artillery_kill_reward", 40.0)
        self.artillery_loss_penalty = getattr(self.config, "artillery_loss_penalty", -40.0)
        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            RelativeAltitudeReward(self.config),
            TeamSpacingReward(self.config),
            EventDrivenReward(self.config),
        ]
        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            OutOfBounds(self.config),
            Timeout(self.config),
        ]
        self._team_event_bonus = {}
        self._last_alive = {}
        self._cumulative_team_score = {}

    def reset(self, env):
        self._team_event_bonus = {"ego": 0.0, "enm": 0.0}
        self._last_alive = {
            "ego": sum(int(env.agents[agent_id].is_alive) for agent_id in env.ego_ids),
            "enm": sum(int(env.agents[agent_id].is_alive) for agent_id in env.enm_ids),
        }
        self._cumulative_team_score = {"ego": 0.0, "enm": 0.0}
        return super().reset(env)

    def step(self, env):
        super().step(env)
        ego_score = self._compute_team_score(env, env.ego_ids)
        enm_score = self._compute_team_score(env, env.enm_ids)
        self._cumulative_team_score["ego"] += ego_score
        self._cumulative_team_score["enm"] += enm_score

        ego_alive = sum(int(env.agents[agent_id].is_alive) for agent_id in env.ego_ids)
        enm_alive = sum(int(env.agents[agent_id].is_alive) for agent_id in env.enm_ids)
        enemy_down = self._last_alive["enm"] - enm_alive
        ally_down = self._last_alive["ego"] - ego_alive
        self._team_event_bonus["ego"] = enemy_down * self.artillery_kill_reward + ally_down * self.artillery_loss_penalty
        self._team_event_bonus["enm"] = ally_down * self.artillery_kill_reward + enemy_down * self.artillery_loss_penalty
        self._last_alive["ego"] = ego_alive
        self._last_alive["enm"] = enm_alive

    def _team_name(self, env, agent_id):
        return "ego" if agent_id in env.ego_ids else "enm"

    def _compute_team_blood_score(self, env, team_ids):
        alive_agents = [env.agents[agent_id] for agent_id in team_ids if env.agents[agent_id].is_alive]
        if len(alive_agents) == 0:
            return 0.0
        return float(np.mean([agent.bloods for agent in alive_agents]) / 100.0)

    def _compute_timeout_value(self, env, team_ids, team_name):
        instant_score = self._compute_team_score(env, team_ids)
        cumulative_score = self._cumulative_team_score[team_name] / max(env.current_step, 1)
        alive_score = sum(int(env.agents[agent_id].is_alive) for agent_id in team_ids) / max(len(team_ids), 1)
        blood_score = self._compute_team_blood_score(env, team_ids)
        return 0.45 * cumulative_score + 0.25 * instant_score + 0.20 * alive_score + 0.10 * blood_score

    def get_reward(self, env, agent_id, info: dict = ...):
        reward, info = super().get_reward(env, agent_id, info=info)
        team_name = self._team_name(env, agent_id)
        reward += self._team_event_bonus.get(team_name, 0.0)
        if isinstance(info, dict) and "episode_summary" in info:
            result = info["episode_summary"]["result"]
            if result == "draw":
                reward += self.timeout_draw_penalty
            elif (result == "win" and team_name == "ego") or (result == "loss" and team_name == "enm"):
                reward += self.timeout_win_reward
            else:
                reward += self.timeout_loss_penalty
        return reward, info

    def get_episode_summary(self, env, info):
        agent_termination = info.get("agent_termination", {})
        ego_alive = sum(int(env.agents[agent_id].is_alive) for agent_id in env.ego_ids)
        enemy_alive = sum(int(env.agents[agent_id].is_alive) for agent_id in env.enm_ids)
        ego_score = self._compute_team_score(env, env.ego_ids) + 0.5 * ego_alive
        enemy_score = self._compute_team_score(env, env.enm_ids) + 0.5 * enemy_alive
        ego_timeout_value = self._compute_timeout_value(env, env.ego_ids, "ego")
        enemy_timeout_value = self._compute_timeout_value(env, env.enm_ids, "enm")
        margin = getattr(self.config, "episode_result_margin", 0.02)

        if enemy_alive == 0 and ego_alive > 0:
            result = "win"
        elif ego_alive == 0 and enemy_alive > 0:
            result = "loss"
        elif ego_timeout_value - enemy_timeout_value > margin:
            result = "win"
        elif enemy_timeout_value - ego_timeout_value > margin:
            result = "loss"
        elif ego_score - enemy_score > margin * 0.5:
            result = "win"
        elif enemy_score - ego_score > margin * 0.5:
            result = "loss"
        else:
            result = "draw"

        info["episode_summary"] = {
            "result": result,
            "steps": env.current_step,
            "ego_alive": ego_alive,
            "enemy_alive": enemy_alive,
            "ego_score": ego_score,
            "enemy_score": enemy_score,
            "ego_timeout_value": ego_timeout_value,
            "enemy_timeout_value": enemy_timeout_value,
            "ego_safety_shield_rate": self.get_safety_shield_rate(env.ego_ids),
            "enemy_safety_shield_rate": self.get_safety_shield_rate(env.enm_ids),
            "ego_end_reasons": sorted({
                reason for agent_id in env.ego_ids for reason in agent_termination.get(agent_id, [])
            }),
            "enemy_end_reasons": sorted({
                reason for agent_id in env.enm_ids for reason in agent_termination.get(agent_id, [])
            }),
        }
        return info


class HierarchicalMultipleCombatTask(MultipleCombatTask):
    
    def __init__(self, config: str):
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu')))
        self.lowlevel_policy.eval()
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        # generate low-level input_obs
        raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12)
        # (1) delta altitude/heading/velocity
        input_obs[0] = self.norm_delta_altitude[action[0]]
        input_obs[1] = self.norm_delta_heading[action[1]]
        input_obs[2] = self.norm_delta_velocity[action[2]]
        # (2) ego info
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.expand_dims(input_obs, axis=0)
        # output low-level action
        _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
        action = _action.detach().cpu().numpy().squeeze(0)
        self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
        # normalize low-level action
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.
        norm_act[1] = action[1] / 20 - 1.
        norm_act[2] = action[2] / 20 - 1.
        norm_act[3] = action[3] / 58 + 0.4
        return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)



class HierarchicalMultipleCombatShootTask(HierarchicalMultipleCombatTask):
    def __init__(self, config: str):
        super().__init__(config)
        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)
        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]
    
    def load_observation_space(self):
        self.obs_length = 9 + self.total_aircraft * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.total_aircraft * self.obs_length,))
    
    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])


    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        # (1) ego info normalization
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)(unit: 5G)
        # (2) relative inof w.r.t partner+enemies state
        offset = 8
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset+1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset+2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset+3] = AO
            norm_obs[offset+4] = TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        # (3) missile info TODO: multiple missile and parnter's missile?
        missile_sim = env.agents[agent_id].check_missile_warning() #
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[offset + 1] = (np.linalg.norm(missile_sim.get_velocity()) - ego_state[9]) / 340
            norm_obs[offset + 2] = (missile_feature[2] - ego_state[2]) / 1000
            norm_obs[offset + 3] = ego_AO
            norm_obs[offset + 4] = ego_TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self._remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self._shoot_action = {agent_id: False for agent_id in env.agents.keys()}
        return super().reset(env)

    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[3] > 0
        return super().normalize_action(env, agent_id, action[:3])

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            # Determine whether can launch missile at the nearest enemy aircraft
            target_list = list(map(lambda x: x.get_position() - agent.get_position(), agent.enemies))
            target_distance = list(map(np.linalg.norm, target_list))
            target_index = np.argmin(target_distance)
            target = target_list[target_index]
            heading = agent.get_velocity()
            distance = target_distance[target_index]
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]

            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self._remaining_missiles[agent_id] > 0 \
                and attack_angle <= self.max_attack_angle and distance <= self.max_attack_distance and shoot_interval >= self.min_attack_interval
            if shoot_flag:
                new_missile_uid = agent_id + str(self._remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[target_index], uid=new_missile_uid))
                self._remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step
