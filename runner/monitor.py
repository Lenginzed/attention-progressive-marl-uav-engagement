import json
from collections import Counter
from pathlib import Path


def extract_episode_summaries(infos):
    if infos is None:
        return []
    summaries = []
    for info in infos:
        if isinstance(info, dict) and "episode_summary" in info:
            summaries.append(info["episode_summary"])
    return summaries


def summarize_episode_summaries(summaries):
    if len(summaries) == 0:
        return {
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "draw_rate": 0.0,
            "avg_episode_steps": 0.0,
            "timeout_rate": 0.0,
            "low_altitude_rate": 0.0,
            "distance_out_rate": 0.0,
            "extreme_state_rate": 0.0,
            "safety_shield_rate": 0.0,
        }

    outcomes = Counter(summary.get("result", "draw") for summary in summaries)
    ego_reasons = Counter()
    avg_steps = 0.0
    avg_safety_shield_rate = 0.0
    for summary in summaries:
        avg_steps += float(summary.get("steps", 0))
        avg_safety_shield_rate += float(summary.get("ego_safety_shield_rate", 0.0))
        for reason in summary.get("ego_end_reasons", []):
            ego_reasons[reason] += 1

    total = float(len(summaries))
    return {
        "win_rate": outcomes["win"] / total,
        "loss_rate": outcomes["loss"] / total,
        "draw_rate": outcomes["draw"] / total,
        "avg_episode_steps": avg_steps / total,
        "timeout_rate": ego_reasons["timeout"] / total,
        "low_altitude_rate": ego_reasons["low_altitude"] / total,
        "distance_out_rate": ego_reasons["out_of_bounds"] / total,
        "extreme_state_rate": ego_reasons["extreme_state"] / total,
        "safety_shield_rate": avg_safety_shield_rate / total,
    }


def build_progress_bar(progress, width):
    progress = min(max(float(progress), 0.0), 1.0)
    completed = min(width, int(round(progress * width)))
    return "[" + "=" * completed + "." * max(width - completed, 0) + "]"


class JSONLMetricsWriter:
    def __init__(self, run_dir, filename="metrics.jsonl", enabled=True):
        self.enabled = enabled
        self.filepath = Path(run_dir) / filename
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload):
        if not self.enabled:
            return
        with self.filepath.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")
