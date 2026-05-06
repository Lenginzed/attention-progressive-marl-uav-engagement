import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pymap3d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_acmi(path):
    trajectories = defaultdict(lambda: {"time": [], "lon": [], "lat": [], "alt": []})
    current_time = 0.0
    with open(path, "r", encoding="utf-8-sig") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                current_time = float(line[1:])
                continue
            if ",T=" not in line:
                continue
            uid, payload = line.split(",T=", 1)
            fields = payload.split(",")[0].split("|")
            if len(fields) < 3:
                continue
            lon, lat, alt = map(float, fields[:3])
            trajectories[uid]["time"].append(current_time)
            trajectories[uid]["lon"].append(lon)
            trajectories[uid]["lat"].append(lat)
            trajectories[uid]["alt"].append(alt)
    return trajectories


def team_style(uid, ego_side):
    is_red_team = uid.startswith("A")
    if ego_side == "blue":
        is_ego = not is_red_team
    else:
        is_ego = is_red_team

    if is_red_team:
        color = "#c43d3d"
        team_name = "Red"
    else:
        color = "#2f6db3"
        team_name = "Blue"
    role = "Ego" if is_ego else "Enemy"
    return {"color": color, "label": f"{uid} ({role}/{team_name})"}


def load_metadata(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_info_box(metadata):
    lines = []
    if metadata.get("matchup_label"):
        lines.append(f"Matchup: {metadata['matchup_label']}")
    if metadata.get("scenario_name"):
        lines.append(f"Scenario: {metadata['scenario_name']}")
    if metadata.get("ego_side"):
        lines.append(f"Ego Side: {metadata['ego_side']}")
    summary = metadata.get("episode_summary", {})
    if summary:
        lines.append(f"Result: {summary.get('ego_result', 'unknown')}")
        lines.append(f"Steps: {summary.get('steps', 0)}")
        lines.append(f"Ego alive: {summary.get('ego_alive', 0)} | Enemy alive: {summary.get('enemy_alive', 0)}")
        ego_reasons = ",".join(summary.get("ego_end_reasons", [])) or "-"
        enemy_reasons = ",".join(summary.get("enemy_end_reasons", [])) or "-"
        lines.append(f"Ego end: {ego_reasons}")
        lines.append(f"Enemy end: {enemy_reasons}")
    return "\n".join(lines)


def classify_endpoint(traj, max_time, tolerance=1.0):
    if not traj["time"]:
        return "End"
    if max_time - traj["time"][-1] > tolerance:
        return "Down"
    return "End"


def convert_to_ned(trajectories, center_lat, center_lon, center_alt):
    converted = {}
    for uid, traj in trajectories.items():
        north, east, down = pymap3d.geodetic2ned(
            traj["lat"], traj["lon"], traj["alt"], center_lat, center_lon, center_alt
        )
        converted[uid] = {
            "time": traj["time"],
            "north": north,
            "east": east,
            "up": -down,
        }
    return converted


def plot_2d_panels(output_path, trajectories_ned, raw_trajectories, metadata, ego_side):
    fig = plt.figure(figsize=(17, 8.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[12, 2.2], wspace=0.24, hspace=0.16)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    footer_ax = fig.add_subplot(grid[1, :])
    footer_ax.axis("off")
    max_time = max(traj["time"][-1] for traj in raw_trajectories.values() if traj["time"])
    legend_handles = []
    legend_labels = []
    for uid, traj in trajectories_ned.items():
        raw_traj = raw_trajectories[uid]
        style = team_style(uid, ego_side)
        endpoint_label = classify_endpoint(raw_traj, max_time)
        line, = axes[0].plot(traj["east"], traj["north"], linewidth=2.5, color=style["color"], label=style["label"])
        axes[0].scatter(traj["east"][0], traj["north"][0], s=90, color=style["color"], marker="o", edgecolors="black", linewidths=0.6, zorder=3)
        axes[0].scatter(traj["east"][-1], traj["north"][-1], s=95, color=style["color"], marker="X", edgecolors="black", linewidths=0.6, zorder=3)
        axes[0].annotate(
            f"{uid} start",
            (traj["east"][0], traj["north"][0]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=style["color"],
        )
        axes[0].annotate(
            f"{uid} {endpoint_label.lower()}",
            (traj["east"][-1], traj["north"][-1]),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=8,
            color=style["color"],
        )

        axes[1].plot(raw_traj["time"], raw_traj["alt"], linewidth=2.5, color=style["color"], label=style["label"])
        axes[1].scatter(raw_traj["time"][0], raw_traj["alt"][0], s=40, color=style["color"], zorder=3)
        axes[1].scatter(raw_traj["time"][-1], raw_traj["alt"][-1], s=50, color=style["color"], marker="X", zorder=3)
        axes[1].annotate(
            f"{uid} {endpoint_label.lower()}",
            (raw_traj["time"][-1], raw_traj["alt"][-1]),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=8,
            color=style["color"],
        )
        legend_handles.append(line)
        legend_labels.append(style["label"])

    axes[0].set_title("Top-Down Air Combat Trajectory")
    axes[0].set_xlabel("East (m)")
    axes[0].set_ylabel("North (m)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_title("Altitude-Time Profile")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=min(4, max(len(legend_labels), 1)),
        bbox_to_anchor=(0.5, 0.955),
        fontsize=9,
        frameon=True,
    )

    info_box = build_info_box(metadata)
    if info_box:
        footer_ax.text(
            0.01,
            0.9,
            info_box,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#444444"),
        )

    fig.suptitle(metadata.get("title", "Annotated Air Combat Trajectory"), fontsize=14)
    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.06, right=0.97)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_3d_panel(output_path, trajectories_ned, raw_trajectories, metadata, ego_side):
    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(2, 1, height_ratios=[12, 2.3], hspace=0.05)
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    footer_ax = fig.add_subplot(grid[1, 0])
    footer_ax.axis("off")
    max_time = max(traj["time"][-1] for traj in raw_trajectories.values() if traj["time"])
    handles = []
    labels = []
    for uid, traj in trajectories_ned.items():
        raw_traj = raw_trajectories[uid]
        style = team_style(uid, ego_side)
        endpoint_label = classify_endpoint(raw_traj, max_time)
        line, = ax.plot(traj["east"], traj["north"], traj["up"], linewidth=2.4, color=style["color"])
        ax.scatter(traj["east"][0], traj["north"][0], traj["up"][0], s=80, color=style["color"], marker="o", edgecolors="black", linewidths=0.5)
        ax.scatter(traj["east"][-1], traj["north"][-1], traj["up"][-1], s=90, color=style["color"], marker="X", edgecolors="black", linewidths=0.5)
        ax.text(traj["east"][0], traj["north"][0], traj["up"][0], f"{uid} start", fontsize=8, color=style["color"])
        ax.text(traj["east"][-1], traj["north"][-1], traj["up"][-1], f"{uid} {endpoint_label.lower()}", fontsize=8, color=style["color"])
        handles.append(line)
        labels.append(style["label"])

    ax.set_title("3D Maneuver Geometry")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Altitude (m)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.view_init(elev=24, azim=-55)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=min(4, max(len(labels), 1)), fontsize=9)

    info_box = build_info_box(metadata)
    if info_box:
        footer_ax.text(
            0.01,
            0.9,
            info_box,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="#444444"),
        )

    fig.suptitle(metadata.get("title", "3D Air Combat Trajectory"), fontsize=14)
    fig.subplots_adjust(top=0.9, bottom=0.06, left=0.05, right=0.97)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to txt.acmi file")
    parser.add_argument("--output", type=str, default=None, help="Output figure path")
    parser.add_argument("--output-3d", type=str, default=None, help="Optional output path for 3D trajectory figure")
    parser.add_argument("--metadata", type=str, default=None, help="Optional metadata json path")
    parser.add_argument("--center-lon", type=float, default=None, help="Reference longitude")
    parser.add_argument("--center-lat", type=float, default=None, help="Reference latitude")
    parser.add_argument("--center-alt", type=float, default=0.0, help="Reference altitude")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name("trajectory_annotated.png")
    trajectories = load_acmi(input_path)
    if len(trajectories) == 0:
        raise RuntimeError(f"No trajectory data found in {input_path}")

    metadata = load_metadata(args.metadata)
    ego_side = metadata.get("ego_side", "red")
    first_uid = next(iter(trajectories))
    center_lon = args.center_lon if args.center_lon is not None else trajectories[first_uid]["lon"][0]
    center_lat = args.center_lat if args.center_lat is not None else trajectories[first_uid]["lat"][0]
    center_alt = args.center_alt
    trajectories_ned = convert_to_ned(trajectories, center_lat, center_lon, center_alt)

    plot_2d_panels(output_path, trajectories_ned, trajectories, metadata, ego_side)
    print(f"Saved annotated figure to {output_path}")

    if args.output_3d:
        output_3d_path = Path(args.output_3d)
        plot_3d_panel(output_3d_path, trajectories_ned, trajectories, metadata, ego_side)
        print(f"Saved 3D figure to {output_3d_path}")


if __name__ == "__main__":
    main()
