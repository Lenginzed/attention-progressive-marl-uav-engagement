import logging
import time
from collections import deque

import numpy as np
import torch

from algorithms.matd3.policy import MATD3Policy
from algorithms.matd3.replay_buffer import MATD3ReplayBuffer
from algorithms.matd3.trainer import MATD3Trainer
from runner.monitor import JSONLMetricsWriter, build_progress_bar, extract_episode_summaries, summarize_episode_summaries


class ShareOffPolicyJSBSimRunner:
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.run_dir = config["run_dir"]

        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = int(self.all_args.num_env_steps)
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.batch_size = self.all_args.batch_size
        self.learning_starts = self.all_args.learning_starts
        self.updates_per_step = self.all_args.updates_per_step
        self.exploration_noise = self.all_args.exploration_noise
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.eval_interval = self.all_args.eval_interval
        self.eval_episodes = self.all_args.eval_episodes
        self.use_eval = self.all_args.use_eval
        self.progress_bar_width = self.all_args.progress_bar_width
        self.episode_stats_window = self.all_args.episode_stats_window
        self.recent_episode_summaries = deque(maxlen=self.episode_stats_window)
        self.recent_step_rewards = deque(maxlen=max(self.episode_stats_window, 128))
        self.metrics_writer = JSONLMetricsWriter(
            self.run_dir,
            filename=self.all_args.metrics_filename,
            enabled=getattr(self.all_args, "export_metrics", True),
        )
        self.last_eval_metrics = {}
        self.total_env_steps = 0

        self.obs_space = self.envs.observation_space
        self.share_obs_space = self.envs.share_observation_space
        self.num_agents = self.envs.num_agents
        self.policy = MATD3Policy(
            self.all_args,
            self.obs_space,
            self.share_obs_space,
            self.envs.action_space,
            num_agents=self.num_agents,
            device=self.device,
        )
        self.trainer = MATD3Trainer(self.all_args, self.policy, self.device)
        self.buffer = MATD3ReplayBuffer(
            self.all_args.offpolicy_buffer_capacity,
            self.num_agents,
            int(np.prod(self.obs_space.shape)),
            int(np.prod(self.share_obs_space.shape)),
            self.policy.action_dim,
        )
        self.model_dir = self.all_args.model_dir
        if self.model_dir is not None:
            self.restore()

    def save(self):
        torch.save(self.policy.actor.state_dict(), str(self.run_dir / "actor_latest.pt"))
        torch.save(self.policy.critic.state_dict(), str(self.run_dir / "critic_latest.pt"))

    def restore(self):
        actor_state = torch.load(str(self.model_dir) + "/actor_latest.pt", map_location=self.device, weights_only=True)
        critic_state = torch.load(str(self.model_dir) + "/critic_latest.pt", map_location=self.device, weights_only=True)
        self.policy.actor.load_state_dict(actor_state)
        self.policy.actor_target.load_state_dict(actor_state)
        self.policy.critic.load_state_dict(critic_state)
        self.policy.critic_target.load_state_dict(critic_state)

    def _collapse_share_obs(self, share_obs):
        return share_obs[:, 0, :]

    def _collapse_rewards(self, rewards):
        return rewards.mean(axis=1)

    def _collapse_dones(self, dones):
        return np.all(dones.squeeze(-1), axis=1, keepdims=True).astype(np.float32)

    def _record_episode_summaries(self, infos):
        summaries = extract_episode_summaries(infos)
        self.recent_episode_summaries.extend(summaries)
        return summaries

    def run(self):
        obs, share_obs = self.envs.reset()
        start = time.time()
        episodes = max(self.num_env_steps // max(self.n_rollout_threads, 1), 1)
        update_info = {"critic_loss": 0.0, "actor_loss": 0.0, "q1_mean": 0.0, "q2_mean": 0.0}
        next_save_step = self.save_interval * max(self.n_rollout_threads, 1)
        next_eval_step = self.eval_interval * max(self.n_rollout_threads, 1)
        next_log_step = self.log_interval * max(self.n_rollout_threads, 1)

        while self.total_env_steps < self.num_env_steps:
            deterministic = self.total_env_steps < self.learning_starts
            if deterministic:
                actions = np.zeros((self.n_rollout_threads, self.num_agents, self.policy.action_dim), dtype=np.float32)
                actions[..., :3] = np.random.uniform(-1.0, 1.0, size=actions[..., :3].shape)
                actions[..., 3] = np.random.uniform(0.4, 0.9, size=actions[..., 3].shape)
            else:
                actions = self.policy.act(obs.reshape(-1, obs.shape[-1]), deterministic=False, noise_std=self.exploration_noise)
                actions = actions.reshape(self.n_rollout_threads, self.num_agents, self.policy.action_dim)

            next_obs, next_share_obs, rewards, dones, infos = self.envs.step(actions)
            self._record_episode_summaries(infos)
            self.recent_step_rewards.extend(self._collapse_rewards(rewards).reshape(-1).tolist())

            self.buffer.insert_batch(
                obs.astype(np.float32),
                self._collapse_share_obs(share_obs).astype(np.float32),
                actions.astype(np.float32),
                self._collapse_rewards(rewards).astype(np.float32),
                self._collapse_dones(dones),
                next_obs.astype(np.float32),
                self._collapse_share_obs(next_share_obs).astype(np.float32),
            )

            obs, share_obs = next_obs, next_share_obs
            self.total_env_steps += self.n_rollout_threads

            if self.buffer.size >= max(self.learning_starts, self.batch_size):
                for _ in range(self.updates_per_step * self.n_rollout_threads):
                    update_info = self.trainer.train(self.buffer, self.batch_size)

            if self.total_env_steps >= next_save_step or self.total_env_steps >= self.num_env_steps:
                self.save()
                next_save_step += self.save_interval * max(self.n_rollout_threads, 1)

            if self.use_eval and (self.total_env_steps >= next_eval_step or self.total_env_steps >= self.num_env_steps):
                self.eval(self.total_env_steps)
                next_eval_step += self.eval_interval * max(self.n_rollout_threads, 1)

            if self.total_env_steps >= next_log_step or self.total_env_steps >= self.num_env_steps:
                progress = min(self.total_env_steps / max(self.num_env_steps, 1), 1.0)
                progress_bar = build_progress_bar(progress, self.progress_bar_width)
                recent_metrics = summarize_episode_summaries(list(self.recent_episode_summaries))
                average_reward = float(np.mean(self.recent_step_rewards)) if self.recent_step_rewards else 0.0
                fps = int(self.total_env_steps / max(time.time() - start, 1e-6))
                logging.info(
                    "\n Scenario {} Algo {} Exp {} total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        self.total_env_steps,
                        self.num_env_steps,
                        fps,
                    )
                )
                logging.info(
                    "{} step {}/{} | reward:{:.3f} | win:{:.3f} loss:{:.3f} draw:{:.3f} | "
                    "end[low:{:.3f} dist:{:.3f} timeout:{:.3f}] | shield:{:.3f} | steps:{:.1f}".format(
                        progress_bar,
                        self.total_env_steps,
                        self.num_env_steps,
                        average_reward,
                        float(recent_metrics["win_rate"]),
                        float(recent_metrics["loss_rate"]),
                        float(recent_metrics["draw_rate"]),
                        float(recent_metrics["low_altitude_rate"]),
                        float(recent_metrics["distance_out_rate"]),
                        float(recent_metrics["timeout_rate"]),
                        float(recent_metrics["safety_shield_rate"]),
                        float(recent_metrics["avg_episode_steps"]),
                    )
                )
                if self.last_eval_metrics:
                    logging.info(
                        " latest eval | reward:{:.3f} | win:{:.3f} loss:{:.3f} draw:{:.3f}".format(
                            float(self.last_eval_metrics.get("eval_average_episode_rewards", 0.0)),
                            float(self.last_eval_metrics.get("eval_win_rate", 0.0)),
                            float(self.last_eval_metrics.get("eval_loss_rate", 0.0)),
                            float(self.last_eval_metrics.get("eval_draw_rate", 0.0)),
                        )
                    )
                record = {
                    "event": "train",
                    "total_num_steps": int(self.total_env_steps),
                    "scenario_name": self.all_args.scenario_name,
                    "average_episode_rewards": average_reward,
                    **{key: float(value) for key, value in recent_metrics.items()},
                    **{key: float(value) for key, value in update_info.items()},
                }
                self.metrics_writer.write(record)
                next_log_step += self.log_interval * max(self.n_rollout_threads, 1)

        self.save()

    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        eval_obs, eval_share_obs = self.eval_envs.reset()
        total_episodes = 0
        episode_rewards = []
        cumulative_rewards = np.zeros((self.n_eval_rollout_threads, 1), dtype=np.float32)
        episode_summaries = []

        while total_episodes < self.eval_episodes:
            actions = self.policy.act(eval_obs.reshape(-1, eval_obs.shape[-1]), deterministic=True)
            actions = actions.reshape(self.n_eval_rollout_threads, self.num_agents, self.policy.action_dim)
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(actions)
            cumulative_rewards += self._collapse_rewards(eval_rewards)
            done_env = self._collapse_dones(eval_dones).squeeze(-1).astype(bool)
            if np.any(done_env):
                episode_rewards.extend(cumulative_rewards[done_env].reshape(-1).tolist())
                cumulative_rewards[done_env] = 0.0
            total_episodes += int(done_env.sum())
            episode_summaries.extend(extract_episode_summaries(eval_infos))

        eval_metrics = summarize_episode_summaries(episode_summaries)
        eval_metrics["eval_average_episode_rewards"] = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        self.last_eval_metrics = {
            "eval_average_episode_rewards": eval_metrics["eval_average_episode_rewards"],
            "eval_win_rate": eval_metrics["win_rate"],
            "eval_loss_rate": eval_metrics["loss_rate"],
            "eval_draw_rate": eval_metrics["draw_rate"],
        }
        self.metrics_writer.write({
            "event": "eval",
            "total_num_steps": int(total_num_steps),
            "scenario_name": self.all_args.scenario_name,
            "eval_average_episode_rewards": float(eval_metrics["eval_average_episode_rewards"]),
            "eval_win_rate": float(eval_metrics["win_rate"]),
            "eval_loss_rate": float(eval_metrics["loss_rate"]),
            "eval_draw_rate": float(eval_metrics["draw_rate"]),
            "eval_low_altitude_rate": float(eval_metrics["low_altitude_rate"]),
            "eval_distance_out_rate": float(eval_metrics["distance_out_rate"]),
            "eval_timeout_rate": float(eval_metrics["timeout_rate"]),
            "eval_safety_shield_rate": float(eval_metrics["safety_shield_rate"]),
            "eval_avg_episode_steps": float(eval_metrics["avg_episode_steps"]),
        })
        logging.info(
            " eval reward:{:.3f} | win:{:.3f} loss:{:.3f} draw:{:.3f} | end[low:{:.3f} dist:{:.3f} timeout:{:.3f}] | shield:{:.3f}".format(
                float(eval_metrics["eval_average_episode_rewards"]),
                float(eval_metrics["win_rate"]),
                float(eval_metrics["loss_rate"]),
                float(eval_metrics["draw_rate"]),
                float(eval_metrics["low_altitude_rate"]),
                float(eval_metrics["distance_out_rate"]),
                float(eval_metrics["timeout_rate"]),
                float(eval_metrics["safety_shield_rate"]),
            )
        )
        logging.info("...End evaluation")
