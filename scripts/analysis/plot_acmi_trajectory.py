import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pymap3d


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to txt.acmi file")
    parser.add_argument("--output", type=str, default=None, help="Output figure path")
    parser.add_argument("--center-lon", type=float, default=None, help="Reference longitude")
    parser.add_argument("--center-lat", type=float, default=None, help="Reference latitude")
    parser.add_argument("--center-alt", type=float, default=0.0, help="Reference altitude")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name("trajectory.png")
    trajectories = load_acmi(input_path)
    if len(trajectories) == 0:
        raise RuntimeError(f"No trajectory data found in {input_path}")

    first_uid = next(iter(trajectories))
    center_lon = args.center_lon if args.center_lon is not None else trajectories[first_uid]["lon"][0]
    center_lat = args.center_lat if args.center_lat is not None else trajectories[first_uid]["lat"][0]
    center_alt = args.center_alt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for uid, traj in trajectories.items():
        north, east, _ = pymap3d.geodetic2ned(
            traj["lat"], traj["lon"], traj["alt"], center_lat, center_lon, center_alt
        )
        axes[0].plot(east, north, linewidth=2, label=uid)
        axes[0].scatter(east[0], north[0], s=24)
        axes[1].plot(traj["time"], traj["alt"], linewidth=2, label=uid)

    axes[0].set_title("Top-Down Trajectory")
    axes[0].set_xlabel("East (m)")
    axes[0].set_ylabel("North (m)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title("Altitude-Time Trajectory")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
