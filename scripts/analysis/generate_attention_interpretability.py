import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import get_config
from algorithms.mappo.ppo_policy import PPOPolicy
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv


def make_policy_args():
    args = get_config().parse_args([])
    args.algorithm_name = "mappo"
    args.use_attention_policy = True
    args.use_progressive_action_discretization = True
    args.use_prior = False
    args.hidden_size = "128 128"
    args.act_hidden_size = "128 128"
    args.recurrent_hidden_size = 128
    args.recurrent_hidden_layers = 1
    args.use_recurrent_policy = True
    args.use_feature_normalization = False
    args.activation_id = 1
    args.gain = 0.01
    args.attention_embed_dim = 128
    args.attention_num_heads = 4
    args.attention_dropout = 0.0
    args.attention_sparse_topk = 2
    args.attention_critic_query_tokens = 2
    args.progressive_action_strides = "10 10 10 7|5 5 5 4|2 2 2 2|1 1 1 1"
    args.progressive_stage_boundaries = "0.0 0.35 0.7 0.9"
    args.lr = 3e-4
    return args


def load_policy(model_dir, obs_space, share_obs_space, act_space, device):
    args = make_policy_args()
    policy = PPOPolicy(args, obs_space, share_obs_space, act_space, device=device)
    actor_state = torch.load(str(Path(model_dir) / "actor_latest.pt"), map_location=device, weights_only=True)
    critic_state = torch.load(str(Path(model_dir) / "critic_latest.pt"), map_location=device, weights_only=True)
    policy.actor.load_state_dict(actor_state)
    policy.critic.load_state_dict(critic_state)
    policy.prep_rollout()
    policy.set_progress(1.0)
    return policy


def reduce_actor_aux(aux):
    reduced = {}
    if aux["self_attention_weights"] is not None:
        reduced["self_attention"] = aux["self_attention_weights"].mean(dim=(0, 1)).cpu().numpy()
    if aux["cross_attention_weights"] is not None:
        reduced["cross_attention"] = aux["cross_attention_weights"].mean(dim=(0, 1, 2)).cpu().numpy()
    reduced["sparse_weights"] = aux["sparse_weights"].mean(dim=0).cpu().numpy()
    reduced["fusion_weights"] = aux["fusion_weights"].cpu().numpy()
    return reduced


def reduce_critic_aux(aux):
    reduced = {}
    if aux["self_attention_weights"] is not None:
        reduced["self_attention"] = aux["self_attention_weights"].mean(dim=(0, 1)).cpu().numpy()
    if aux["cross_attention_weights"] is not None:
        reduced["cross_attention"] = aux["cross_attention_weights"].mean(dim=(0, 1, 2)).cpu().numpy()
    reduced["sparse_weights"] = aux["sparse_weights"].mean(dim=0).cpu().numpy()
    reduced["fusion_weights"] = aux["fusion_weights"].cpu().numpy()
    return reduced


def average_collection(items, key):
    values = [item[key] for item in items if key in item]
    if not values:
        return None
    return np.mean(np.stack(values, axis=0), axis=0)


def default_actor_labels(token_dims):
    if token_dims == [9, 6, 6, 6]:
        return ["Ego", "Teammate", "Enemy 1", "Enemy 2"]
    return [f"Token {idx + 1}" for idx in range(len(token_dims))]


def default_critic_labels(token_dims):
    if len(token_dims) == 4:
        return ["Red 1", "Red 2", "Blue 1", "Blue 2"]
    return [f"Token {idx + 1}" for idx in range(len(token_dims))]


def plot_interpretability(output_path, actor_summary, critic_summary):
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.4))
    actor_labels = actor_summary["token_labels"]
    critic_labels = critic_summary["token_labels"]

    im0 = axes[0, 0].imshow(actor_summary["self_attention"], cmap="YlGnBu", vmin=0.0, vmax=1.0)
    axes[0, 0].set_xticks(range(len(actor_labels)), actor_labels, rotation=25, ha="right")
    axes[0, 0].set_yticks(range(len(actor_labels)), actor_labels)
    axes[0, 0].set_title("Actor self-attention")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    axes[0, 1].bar(range(len(actor_summary["cross_labels"])), actor_summary["cross_attention"], color="#4472c4")
    axes[0, 1].set_xticks(range(len(actor_summary["cross_labels"])), actor_summary["cross_labels"], rotation=20, ha="right")
    axes[0, 1].set_ylim(0.0, max(1.0, actor_summary["cross_attention"].max() * 1.15))
    axes[0, 1].set_title("Actor cross-attention")
    axes[0, 1].set_ylabel("Average weight")

    axes[0, 2].bar(range(len(actor_labels)), actor_summary["sparse_weights"], color="#70ad47")
    axes[0, 2].set_xticks(range(len(actor_labels)), actor_labels, rotation=25, ha="right")
    axes[0, 2].set_ylim(0.0, max(1.0, actor_summary["sparse_weights"].max() * 1.15))
    axes[0, 2].set_title("Actor sparse-token weights")

    fusion_labels = ["Self", "Cross", "Sparse"]
    axes[0, 3].bar(range(3), actor_summary["fusion_weights"], color=["#5b9bd5", "#ed7d31", "#70ad47"])
    axes[0, 3].set_xticks(range(3), fusion_labels)
    axes[0, 3].set_ylim(0.0, 1.0)
    axes[0, 3].set_title("Actor fusion weights")

    im1 = axes[1, 0].imshow(critic_summary["self_attention"], cmap="YlOrRd", vmin=0.0, vmax=1.0)
    axes[1, 0].set_xticks(range(len(critic_labels)), critic_labels, rotation=25, ha="right")
    axes[1, 0].set_yticks(range(len(critic_labels)), critic_labels)
    axes[1, 0].set_title("Critic self-attention")
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].bar(range(len(critic_summary["cross_labels"])), critic_summary["cross_attention"], color="#c55a11")
    axes[1, 1].set_xticks(range(len(critic_summary["cross_labels"])), critic_summary["cross_labels"], rotation=20, ha="right")
    axes[1, 1].set_ylim(0.0, max(1.0, critic_summary["cross_attention"].max() * 1.15))
    axes[1, 1].set_title("Critic cross-attention")
    axes[1, 1].set_ylabel("Average weight")

    axes[1, 2].bar(range(len(critic_labels)), critic_summary["sparse_weights"], color="#a5a5a5")
    axes[1, 2].set_xticks(range(len(critic_labels)), critic_labels, rotation=25, ha="right")
    axes[1, 2].set_ylim(0.0, max(1.0, critic_summary["sparse_weights"].max() * 1.15))
    axes[1, 2].set_title("Critic sparse-token weights")

    axes[1, 3].bar(range(3), critic_summary["fusion_weights"], color=["#5b9bd5", "#ed7d31", "#70ad47"])
    axes[1, 3].set_xticks(range(3), fusion_labels)
    axes[1, 3].set_ylim(0.0, 1.0)
    axes[1, 3].set_title("Critic fusion weights")

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate interpretability figures for the attention-enhanced MAPPO policy.")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--scenario-name", type=str, default="2v2/ResearchHandcrafted/Stage3")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "attention_interpretability.png"
    summary_path = output_dir / "attention_interpretability.json"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = MultipleCombatEnv(args.scenario_name)
    policy = load_policy(args.model_dir, env.observation_space, env.share_observation_space, env.action_space, device)

    actor_records = []
    critic_records = []
    successful_episodes = 0
    sample_steps = 0
    actor_token_dims = None
    critic_token_dims = None
    actor_query_token_count = None
    critic_query_token_count = None

    for episode_idx in range(args.episodes):
        env.seed(args.seed + episode_idx)
        obs, share_obs = env.reset()
        rnn_states = np.zeros((obs.shape[0], 1, 128), dtype=np.float32)
        masks = np.ones((obs.shape[0], 1), dtype=np.float32)
        episode_actor_records = []
        episode_critic_records = []

        while True:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            share_obs_tensor = torch.as_tensor(share_obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                _, actor_aux = policy.actor.base(obs_tensor, return_aux=True)
                _, critic_aux = policy.critic.base(share_obs_tensor, return_aux=True)
                actions, rnn_states_tensor = policy.act(obs, rnn_states, masks, deterministic=True)
                rnn_states = rnn_states_tensor.detach().cpu().numpy()

            actor_token_dims = actor_aux["token_dims"]
            critic_token_dims = critic_aux["token_dims"]
            actor_query_token_count = actor_aux["query_token_count"]
            critic_query_token_count = critic_aux["query_token_count"]
            episode_actor_records.append(reduce_actor_aux(actor_aux))
            episode_critic_records.append(reduce_critic_aux(critic_aux))

            obs, share_obs, rewards, dones, info = env.step(actions.detach().cpu().numpy())
            done_flags = dones.squeeze(-1).astype(bool)
            masks = np.ones_like(masks, dtype=np.float32)
            masks[done_flags] = 0.0
            if done_flags.all():
                break

        if info["episode_summary"]["result"] == "win":
            successful_episodes += 1
            actor_records.extend(episode_actor_records)
            critic_records.extend(episode_critic_records)
            sample_steps += len(episode_actor_records)

    env.close()

    if not actor_records or not critic_records:
        raise RuntimeError("No successful episodes were collected for attention interpretability analysis.")

    actor_labels = default_actor_labels(actor_token_dims)
    critic_labels = default_critic_labels(critic_token_dims)
    actor_summary = {
        "token_dims": actor_token_dims,
        "query_token_count": actor_query_token_count,
        "token_labels": actor_labels,
        "cross_labels": actor_labels[actor_query_token_count:],
        "self_attention": average_collection(actor_records, "self_attention"),
        "cross_attention": average_collection(actor_records, "cross_attention"),
        "sparse_weights": average_collection(actor_records, "sparse_weights"),
        "fusion_weights": average_collection(actor_records, "fusion_weights"),
    }
    critic_summary = {
        "token_dims": critic_token_dims,
        "query_token_count": critic_query_token_count,
        "token_labels": critic_labels,
        "cross_labels": critic_labels[critic_query_token_count:],
        "self_attention": average_collection(critic_records, "self_attention"),
        "cross_attention": average_collection(critic_records, "cross_attention"),
        "sparse_weights": average_collection(critic_records, "sparse_weights"),
        "fusion_weights": average_collection(critic_records, "fusion_weights"),
    }

    plot_interpretability(figure_path, actor_summary, critic_summary)

    payload = {
        "scenario_name": args.scenario_name,
        "episodes_requested": args.episodes,
        "successful_episodes": successful_episodes,
        "sample_steps": sample_steps,
        "actor": {
            "token_dims": actor_summary["token_dims"],
            "query_token_count": actor_summary["query_token_count"],
            "token_labels": actor_summary["token_labels"],
            "cross_labels": actor_summary["cross_labels"],
            "self_attention": actor_summary["self_attention"].tolist(),
            "cross_attention": actor_summary["cross_attention"].tolist(),
            "sparse_weights": actor_summary["sparse_weights"].tolist(),
            "fusion_weights": actor_summary["fusion_weights"].tolist(),
        },
        "critic": {
            "token_dims": critic_summary["token_dims"],
            "query_token_count": critic_summary["query_token_count"],
            "token_labels": critic_summary["token_labels"],
            "cross_labels": critic_summary["cross_labels"],
            "self_attention": critic_summary["self_attention"].tolist(),
            "cross_attention": critic_summary["cross_attention"].tolist(),
            "sparse_weights": critic_summary["sparse_weights"].tolist(),
            "fusion_weights": critic_summary["fusion_weights"].tolist(),
        },
    }
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)

    print(f"Saved attention interpretability figure to {figure_path}")
    print(f"Saved attention interpretability summary to {summary_path}")


if __name__ == "__main__":
    main()
