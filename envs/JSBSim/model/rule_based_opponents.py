import numpy as np
import torch

from ..core.catalog import Catalog as c
from ..utils.utils import get_root_dir, in_range_rad
from .baseline_actor import BaselineActor


class BaselineLowLevelController:
    def __init__(self):
        self.model_path = get_root_dir() + "/model/baseline_model.pt"
        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device("cpu"), weights_only=True))
        self.actor.eval()
        self.rnn_states = {}
        self.state_var = [
            c.delta_altitude,
            c.delta_heading,
            c.delta_velocities_u,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.velocities_u_mps,
            c.velocities_v_mps,
            c.velocities_w_mps,
            c.velocities_vc_mps,
            c.position_h_sl_m,
        ]

    def reset(self, agent_ids):
        self.rnn_states = {agent_id: np.zeros((1, 1, 128), dtype=np.float32) for agent_id in agent_ids}

    def _build_baseline_observation(self, sim, delta_altitude, delta_heading, delta_velocity):
        obs = sim.get_property_values(self.state_var)
        norm_obs = np.zeros(12, dtype=np.float32)
        norm_obs[0] = delta_altitude / 1000.0
        norm_obs[1] = in_range_rad(delta_heading)
        norm_obs[2] = delta_velocity / 340.0
        norm_obs[3] = obs[9] / 5000.0
        norm_obs[4] = np.sin(obs[3])
        norm_obs[5] = np.cos(obs[3])
        norm_obs[6] = np.sin(obs[4])
        norm_obs[7] = np.cos(obs[4])
        norm_obs[8] = obs[5] / 340.0
        norm_obs[9] = obs[6] / 340.0
        norm_obs[10] = obs[7] / 340.0
        norm_obs[11] = obs[8] / 340.0
        return np.expand_dims(norm_obs, axis=0)

    def _normalize_discrete_action(self, action):
        norm_act = np.zeros(4, dtype=np.float32)
        norm_act[0] = action[0] / 20.0 - 1.0
        norm_act[1] = action[1] / 20.0 - 1.0
        norm_act[2] = action[2] / 20.0 - 1.0
        norm_act[3] = action[3] / 58.0 + 0.4
        return norm_act

    def get_action(self, sim, delta_altitude, delta_heading, delta_velocity):
        obs = self._build_baseline_observation(sim, delta_altitude, delta_heading, delta_velocity)
        action, next_state = self.actor(obs, self.rnn_states[sim.uid])
        self.rnn_states[sim.uid] = next_state.detach().cpu().numpy()
        action = action.detach().cpu().numpy().squeeze(0)
        return self._normalize_discrete_action(action)


class HandcraftedOpponentTeam:
    def __init__(self, level: str):
        self.level = level
        self.low_level = BaselineLowLevelController()
        self.cached_step = -1
        self.cached_actions = {}

    def reset(self, env):
        self.low_level.reset(env.enm_ids)
        self.cached_step = -1
        self.cached_actions = {}

    def get_action(self, env, agent_id):
        if self.cached_step != env.current_step:
            self.cached_actions = self._compute_team_actions(env)
            self.cached_step = env.current_step
        return self.cached_actions[agent_id]

    def _compute_team_actions(self, env):
        if self.level == "easy":
            return self._compute_easy_actions(env)
        if self.level == "medium":
            return self._compute_medium_actions(env)
        if self.level == "hard":
            return self._compute_hard_actions(env)
        raise NotImplementedError(f"Unknown handcrafted opponent level: {self.level}")

    def _compute_easy_actions(self, env):
        assignments = self._independent_assignments(env)
        params = {
            "lead_factor": 0.2,
            "spacing_target": 3000.0,
            "spacing_min": 900.0,
            "boundary_ratio": 0.82,
            "speed_base": 235.0,
            "speed_far_bonus": 10.0,
            "altitude_gain": 0.6,
            "flank_angle": 0.0,
            "altitude_bias": 0.0,
        }
        return self._build_actions_from_assignments(env, assignments, params)

    def _compute_medium_actions(self, env):
        assignments = self._greedy_unique_assignments(env)
        params = {
            "lead_factor": 0.55,
            "spacing_target": 2200.0,
            "spacing_min": 700.0,
            "boundary_ratio": 0.88,
            "speed_base": 248.0,
            "speed_far_bonus": 18.0,
            "altitude_gain": 0.8,
            "flank_angle": np.deg2rad(10.0),
            "altitude_bias": 150.0,
        }
        return self._build_actions_from_assignments(env, assignments, params)

    def _compute_hard_actions(self, env):
        assignments = self._hard_focus_assignments(env)
        params = {
            "lead_factor": 0.9,
            "spacing_target": 1800.0,
            "spacing_min": 650.0,
            "boundary_ratio": 0.92,
            "speed_base": 258.0,
            "speed_far_bonus": 24.0,
            "altitude_gain": 1.0,
            "flank_angle": np.deg2rad(24.0),
            "altitude_bias": 250.0,
        }
        return self._build_actions_from_assignments(env, assignments, params)

    def _independent_assignments(self, env):
        assignments = {}
        for agent_id in env.enm_ids:
            assignments[agent_id] = {
                "target_id": self._nearest_alive_target(env, agent_id),
                "role_sign": 0.0,
                "focus_same_target": False,
            }
        return assignments

    def _greedy_unique_assignments(self, env):
        alive_enemies = [agent_id for agent_id in env.enm_ids if env.agents[agent_id].is_alive]
        alive_targets = [agent_id for agent_id in env.ego_ids if env.agents[agent_id].is_alive]
        assignments = {}
        if len(alive_enemies) == 0:
            for agent_id in env.enm_ids:
                assignments[agent_id] = {"target_id": env.ego_ids[0], "role_sign": 0.0, "focus_same_target": False}
            return assignments
        if len(alive_targets) == 0:
            for agent_id in env.enm_ids:
                assignments[agent_id] = {"target_id": env.ego_ids[0], "role_sign": 0.0, "focus_same_target": False}
            return assignments

        unassigned_targets = set(alive_targets)
        for idx, agent_id in enumerate(alive_enemies):
            if len(unassigned_targets) == 0:
                unassigned_targets = set(alive_targets)
            best_target = min(
                unassigned_targets,
                key=lambda target_id: np.linalg.norm(
                    env.agents[target_id].get_position()[:2] - env.agents[agent_id].get_position()[:2]
                ),
            )
            assignments[agent_id] = {
                "target_id": best_target,
                "role_sign": -1.0 if idx % 2 == 0 else 1.0,
                "focus_same_target": False,
            }
            unassigned_targets.discard(best_target)

        for agent_id in env.enm_ids:
            if agent_id not in assignments:
                assignments[agent_id] = {"target_id": alive_targets[0], "role_sign": 0.0, "focus_same_target": False}
        return assignments

    def _hard_focus_assignments(self, env):
        alive_enemies = [agent_id for agent_id in env.enm_ids if env.agents[agent_id].is_alive]
        alive_targets = [agent_id for agent_id in env.ego_ids if env.agents[agent_id].is_alive]
        assignments = {}
        if len(alive_enemies) == 0:
            for agent_id in env.enm_ids:
                assignments[agent_id] = {"target_id": env.ego_ids[0], "role_sign": 0.0, "focus_same_target": True}
            return assignments
        if len(alive_targets) == 0:
            for agent_id in env.enm_ids:
                assignments[agent_id] = {"target_id": env.ego_ids[0], "role_sign": 0.0, "focus_same_target": True}
            return assignments

        enemy_center = np.mean([env.agents[agent_id].get_position()[:2] for agent_id in alive_enemies], axis=0)
        primary_target = min(
            alive_targets,
            key=lambda target_id: np.linalg.norm(env.agents[target_id].get_position()[:2] - enemy_center),
        )

        if len(alive_enemies) == 1:
            assignments[alive_enemies[0]] = {
                "target_id": primary_target,
                "role_sign": 0.0,
                "focus_same_target": True,
            }
        elif len(alive_enemies) >= 2:
            distances = {
                agent_id: np.linalg.norm(
                    env.agents[primary_target].get_position()[:2] - env.agents[agent_id].get_position()[:2]
                )
                for agent_id in alive_enemies
            }
            lead_agent = min(alive_enemies, key=lambda agent_id: distances[agent_id])
            support_agents = [agent_id for agent_id in alive_enemies if agent_id != lead_agent]
            assignments[lead_agent] = {
                "target_id": primary_target,
                "role_sign": 0.0,
                "focus_same_target": True,
            }
            for idx, agent_id in enumerate(support_agents):
                assignments[agent_id] = {
                    "target_id": primary_target,
                    "role_sign": -1.0 if idx % 2 == 0 else 1.0,
                    "focus_same_target": True,
                }

        for agent_id in env.enm_ids:
            if agent_id not in assignments:
                assignments[agent_id] = {
                    "target_id": primary_target,
                    "role_sign": 0.0,
                    "focus_same_target": True,
                }
        return assignments

    def _build_actions_from_assignments(self, env, assignments, params):
        max_distance = getattr(env.config, "max_distance_from_center_m", 25000.0)
        actions = {}
        for agent_id in env.enm_ids:
            sim = env.agents[agent_id]
            assignment = assignments[agent_id]
            if not sim.is_alive:
                actions[agent_id] = np.array([0.0, 0.0, 0.0, 0.4], dtype=np.float32)
                continue

            target = env.agents[assignment["target_id"]]
            wingman = self._get_wingman(env, agent_id)
            target_vector = self._predict_intercept_vector(sim, target, params["lead_factor"])
            desired_heading = np.arctan2(target_vector[1], target_vector[0])

            if wingman is not None and wingman.is_alive:
                desired_heading += self._formation_bias(
                    sim,
                    wingman,
                    assignment["role_sign"],
                    params["spacing_target"],
                    params["spacing_min"],
                )

            desired_heading += self._boundary_bias(sim, max_distance, params["boundary_ratio"])
            desired_heading += assignment["role_sign"] * params["flank_angle"]

            delta_heading = in_range_rad(desired_heading - sim.get_property_value(c.attitude_heading_true_rad))
            delta_altitude = self._compute_altitude_delta(sim, target, assignment["role_sign"], params)
            delta_velocity = self._compute_velocity_delta(sim, target, params)

            actions[agent_id] = self.low_level.get_action(sim, delta_altitude, delta_heading, delta_velocity)
        return actions

    def _nearest_alive_target(self, env, agent_id):
        sim = env.agents[agent_id]
        alive_targets = [target_id for target_id in env.ego_ids if env.agents[target_id].is_alive]
        if len(alive_targets) == 0:
            alive_targets = list(env.ego_ids)
        return min(
            alive_targets,
            key=lambda target_id: np.linalg.norm(
                env.agents[target_id].get_position()[:2] - sim.get_position()[:2]
            ),
        )

    def _get_wingman(self, env, agent_id):
        wingmen = [candidate for candidate in env.enm_ids if candidate != agent_id]
        if len(wingmen) == 0:
            return None
        return env.agents[wingmen[0]]

    def _predict_intercept_vector(self, sim, target, lead_factor):
        relative = target.get_position()[:2] - sim.get_position()[:2]
        relative_velocity = target.get_velocity()[:2] - sim.get_velocity()[:2]
        distance = max(np.linalg.norm(relative), 1.0)
        speed = max(np.linalg.norm(sim.get_velocity()[:2]), 1.0)
        lead_horizon = min(distance / speed, 8.0) * lead_factor
        intercept = relative + lead_horizon * relative_velocity
        if np.linalg.norm(intercept) < 1e-6:
            intercept = relative
        return intercept

    def _formation_bias(self, sim, wingman, role_sign, spacing_target, spacing_min):
        wingman_vec = wingman.get_position()[:2] - sim.get_position()[:2]
        spacing = np.linalg.norm(wingman_vec)
        if spacing < max(spacing_min, 1.0):
            separation_heading = np.arctan2(-wingman_vec[1], -wingman_vec[0])
            return 0.45 * in_range_rad(separation_heading - sim.get_property_value(c.attitude_heading_true_rad))
        if spacing > spacing_target * 1.6:
            regroup_heading = np.arctan2(wingman_vec[1], wingman_vec[0])
            return 0.15 * in_range_rad(regroup_heading - sim.get_property_value(c.attitude_heading_true_rad))
        if role_sign == 0.0:
            return 0.0
        return role_sign * np.deg2rad(5.0)

    def _boundary_bias(self, sim, max_distance, boundary_ratio):
        position = sim.get_position()[:2]
        distance = np.linalg.norm(position)
        if distance < max_distance * boundary_ratio:
            return 0.0
        center_heading = np.arctan2(-position[1], -position[0])
        return 0.6 * in_range_rad(center_heading - sim.get_property_value(c.attitude_heading_true_rad))

    def _compute_altitude_delta(self, sim, target, role_sign, params):
        target_altitude = target.get_position()[2] + role_sign * params["altitude_bias"]
        return params["altitude_gain"] * (target_altitude - sim.get_position()[2])

    def _compute_velocity_delta(self, sim, target, params):
        distance = np.linalg.norm(target.get_position()[:2] - sim.get_position()[:2])
        desired_speed = params["speed_base"]
        if distance > 5000.0:
            desired_speed += params["speed_far_bonus"]
        if distance < 1800.0:
            desired_speed -= 12.0
        return desired_speed - sim.get_property_value(c.velocities_u_mps)


def build_handcrafted_opponent(level: str):
    return HandcraftedOpponentTeam(level=level)
