#!/usr/bin/env python3
"""Replay recorded DRL actions through SimCon and optionally show start/goal spheres in 3D.

Loads CSV actions ``[-1, 1]``, runs the quadcopter integrator, plots time series, and
runs ``sameAxisAnimation``. Initial pose is ``(start_xy, HOVER_ALTITUDE)`` when
``--goals-json`` is set, or when the CSV lives under ``.../steps/`` and a sibling
``summary.csv`` lists ``start_pos`` / ``goal_pos`` for that episode; otherwise SimCon
defaults (origin at hover height). Spheres use ``GOAL_THRESHOLD`` from ``parameter.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from parameter import (
    DRL_STEP_DURATION,
    GOAL_THRESHOLD,
    HOVER_ALTITUDE,
    PHYSICS_TS,
    SPEED_ANGULAR_MAX,
    SPEED_LINEAR_MAX,
    SPEED_LINEAR_Y_MAX,
)


import utils as _root_utils
normalize_angle = _root_utils.normalize_angle
del sys.modules['utils']

# ------------------------------------------------------------------
# SimCon on path (must follow root imports — SimCon also has a 'utils' package)
# ------------------------------------------------------------------

SIM_ROOT = Path(__file__).resolve().parent / "Quadcopter_SimCon" / "Simulation"
sys.path.insert(0, str(SIM_ROOT))

import config as sim_config
import utils as sim_utils
from ctrl import Control
from quadFiles.quad import Quadcopter
from utils.windModel import Wind


# ------------------------------------------------------------------
# Trajectory adapter (matches QuadNavEnv cmd-vel mode)
# ------------------------------------------------------------------


class CmdVelTrajectory:
    """Velocity-command trajectory: ``xy_vel_z_pos`` with fixed hover height."""

    def __init__(self, hover_altitude: float = HOVER_ALTITUDE) -> None:
        self.ctrlType = "xy_vel_z_pos"
        self.yawType = 1
        self.xyzType = 0
        self.sDes = np.zeros(19)
        self.sDes[2] = hover_altitude
        self.des_yaw = 0.0
        self.wps = np.array([[0.0, 0.0, hover_altitude]])

    def set_cmd_vel(self, linear_vel: float, lateral_vel: float, angular_vel: float, yaw: float, dt: float) -> None:
        self.sDes[3] = linear_vel * np.cos(yaw) + lateral_vel * (-np.sin(yaw))
        self.sDes[4] = linear_vel * np.sin(yaw) + lateral_vel * np.cos(yaw)
        self.sDes[5] = 0.0
        self.des_yaw += angular_vel * dt
        self.des_yaw = normalize_angle(self.des_yaw)
        self.sDes[14] = self.des_yaw


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------


def load_goals_pose_3d(json_path: str, goal_seed: int, hover_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Load start pose and one RNG-chosen goal from goals JSON; both as ``(3,)`` with ``hover_z``."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    start = np.asarray(data["start_pose"], dtype=np.float64)
    goals = [np.asarray(g, dtype=np.float64) for g in data["goal_pose_list"]]
    if not goals:
        raise ValueError(f"goal_pose_list is empty: {json_path}")
    rng = random.Random(goal_seed)
    g = goals[rng.randrange(len(goals))]
    start_3d = np.array([start[0], start[1], float(hover_z)], dtype=np.float64)
    goal_3d = np.array([g[0], g[1], float(hover_z)], dtype=np.float64)
    return start_3d, goal_3d


def load_trajectory_csv(csv_path: str) -> np.ndarray:
    """Load normalized actions from CSV (2-col legacy or 3-col format)."""
    actions: list[list[float]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for row in reader:
            if "lateral_velocity_y" in fields:
                actions.append([
                    float(row["linear_velocity_x"]),
                    float(row["lateral_velocity_y"]),
                    float(row["angular_velocity"]),
                ])
            else:
                actions.append([
                    float(row["linear_velocity"]),
                    0.0,
                    float(row["angular_velocity"]),
                ])
    return np.array(actions, dtype=np.float64)


# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------


def set_quad_world_pose(quad: Quadcopter, xyz: np.ndarray) -> None:
    """Set position and integrator IC (same pattern as ``QuadNavEnv.init_quad_at``)."""
    quad.state[0] = float(xyz[0])
    quad.state[1] = float(xyz[1])
    quad.state[2] = float(xyz[2])
    quad.pos = quad.state[0:3].copy()
    quad.integrator.set_initial_value(np.asarray(quad.state, dtype=float), 0.0)


def lookup_start_goal_from_summary(steps_csv: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Read start/goal positions from the summary.csv next to the steps/ directory.

    Expects ``steps_csv`` at ``<eval_dir>/steps/<NNN>_<map>.csv``.
    Returns ``(start_3d, goal_3d)`` at ``HOVER_ALTITUDE``, or ``(None, None)`` if not found.
    """
    summary_path = steps_csv.parent.parent / "summary.csv"
    if not summary_path.is_file():
        return None, None
    ep_num = int(steps_csv.stem.split("_")[0])
    with open(summary_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "start_pos" not in (reader.fieldnames or []):
            return None, None
        for row in reader:
            if int(row["episode"]) == ep_num:
                sx, sy = (float(v) for v in row["start_pos"].split())
                gx, gy = (float(v) for v in row["goal_pos"].split())
                start_3d = np.array([sx, sy, float(HOVER_ALTITUDE)], dtype=np.float64)
                goal_3d = np.array([gx, gy, float(HOVER_ALTITUDE)], dtype=np.float64)
                return start_3d, goal_3d
    return None, None


def simulate_trajectory_from_csv(
    csv_path: str,
    save_output: bool = True,
    goals_json: Optional[str] = None,
    goal_seed: int = 0,
    start_pos_3d: Optional[np.ndarray] = None,
    goal_pos_3d: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Integrate SimCon with actions from ``csv_path``; optional goals for 3D markers."""
    t0 = time.time()

    actions = load_trajectory_csv(csv_path)
    print(f"Loaded {len(actions)} actions from {csv_path}")

    csv_p = Path(csv_path)
    if start_pos_3d is None or goal_pos_3d is None:
        if goals_json:
            start_pos_3d, goal_pos_3d = load_goals_pose_3d(goals_json, goal_seed, HOVER_ALTITUDE)
        else:
            sp, gp = lookup_start_goal_from_summary(csv_p)
            if sp is not None:
                start_pos_3d, goal_pos_3d = sp, gp
                print(
                    f"Start/goal from summary.csv: start=({start_pos_3d[0]:.2f}, {start_pos_3d[1]:.2f}) "
                    f"goal=({goal_pos_3d[0]:.2f}, {goal_pos_3d[1]:.2f})"
                )

    ti = 0
    ts = PHYSICS_TS
    steps_per_action = int(DRL_STEP_DURATION / PHYSICS_TS)

    csv_duration = len(actions) * DRL_STEP_DURATION
    tf = max(60.0, csv_duration)
    total_steps = int(tf / ts) + 1

    quad = Quadcopter(ti)
    traj = CmdVelTrajectory(hover_altitude=HOVER_ALTITUDE)
    if start_pos_3d is not None:
        set_quad_world_pose(quad, start_pos_3d)
        traj.sDes[0] = float(start_pos_3d[0])
        traj.sDes[1] = float(start_pos_3d[1])
        traj.sDes[2] = float(start_pos_3d[2])
        traj.wps = np.array([[start_pos_3d[0], start_pos_3d[1], start_pos_3d[2]]])
        print(
            f"Initial pose (world): x={start_pos_3d[0]:.3f} y={start_pos_3d[1]:.3f} z={start_pos_3d[2]:.3f} m"
        )
    ctrl = Control(quad, traj.yawType)
    wind = Wind("None", 2.0, 90, -15)

    ctrl.controller(traj, quad, traj.sDes, ts)

    n = total_steps
    t_all = np.zeros(n)
    pos_all = np.zeros((n, len(quad.pos)))
    vel_all = np.zeros((n, len(quad.vel)))
    quat_all = np.zeros((n, len(quad.quat)))
    omega_all = np.zeros((n, len(quad.omega)))
    euler_all = np.zeros((n, len(quad.euler)))
    sDes_traj_all = np.zeros((n, len(traj.sDes)))
    sDes_calc_all = np.zeros((n, len(ctrl.sDesCalc)))
    w_cmd_all = np.zeros((n, len(ctrl.w_cmd)))
    wmotor_all = np.zeros((n, len(quad.wMotor)))
    thr_all = np.zeros((n, len(quad.thr)))
    tor_all = np.zeros((n, len(quad.tor)))

    t_all[0] = ti
    pos_all[0, :] = quad.pos
    vel_all[0, :] = quad.vel
    quat_all[0, :] = quad.quat
    omega_all[0, :] = quad.omega
    euler_all[0, :] = quad.euler
    sDes_traj_all[0, :] = traj.sDes
    sDes_calc_all[0, :] = ctrl.sDesCalc
    w_cmd_all[0, :] = ctrl.w_cmd
    wmotor_all[0, :] = quad.wMotor
    thr_all[0, :] = quad.thr
    tor_all[0, :] = quad.tor

    t = ti
    action_idx = 0
    i = 1

    while round(t, 3) < tf and i < n:
        if action_idx < len(actions):
            action = actions[action_idx]
            linear_vel = (action[0] + 1.0) / 2.0 * SPEED_LINEAR_MAX
            lateral_vel = action[1] * SPEED_LINEAR_Y_MAX
            angular_vel = action[2] * SPEED_ANGULAR_MAX
            action_idx += 1
        else:
            linear_vel = 0.0
            lateral_vel = 0.0
            angular_vel = 0.0

        for _ in range(steps_per_action):
            yaw = quad.euler[2]
            traj.set_cmd_vel(linear_vel, lateral_vel, angular_vel, yaw, ts)
            ctrl.controller(traj, quad, traj.sDes, ts)
            quad.update(t, ts, ctrl.w_cmd, wind)
            t += ts

            if i < n:
                t_all[i] = t
                pos_all[i, :] = quad.pos
                vel_all[i, :] = quad.vel
                quat_all[i, :] = quad.quat
                omega_all[i, :] = quad.omega
                euler_all[i, :] = quad.euler
                sDes_traj_all[i, :] = traj.sDes
                sDes_calc_all[i, :] = ctrl.sDesCalc
                w_cmd_all[i, :] = ctrl.w_cmd
                wmotor_all[i, :] = quad.wMotor
                thr_all[i, :] = quad.thr
                tor_all[i, :] = quad.tor

            i += 1

            if round(t, 3) >= tf or i >= n:
                break

    t_all = t_all[:i]
    pos_all = pos_all[:i, :]
    vel_all = vel_all[:i, :]
    quat_all = quat_all[:i, :]
    omega_all = omega_all[:i, :]
    euler_all = euler_all[:i, :]
    sDes_traj_all = sDes_traj_all[:i, :]
    sDes_calc_all = sDes_calc_all[:i, :]
    w_cmd_all = w_cmd_all[:i, :]
    wmotor_all = wmotor_all[:i, :]
    thr_all = thr_all[:i, :]
    tor_all = tor_all[:i, :]

    print(f"Simulated {t:.2f}s in {time.time() - t0:.6f}s.")

    results: dict[str, Any] = {
        "t_all": t_all,
        "pos_all": pos_all,
        "vel_all": vel_all,
        "quat_all": quat_all,
        "omega_all": omega_all,
        "euler_all": euler_all,
        "sDes_traj_all": sDes_traj_all,
        "sDes_calc_all": sDes_calc_all,
        "w_cmd_all": w_cmd_all,
        "wmotor_all": wmotor_all,
        "thr_all": thr_all,
        "tor_all": tor_all,
        "quad": quad,
        "actions": actions,
        "traj": traj,
        "start_pos_3d": start_pos_3d,
        "goal_pos_3d": goal_pos_3d,
    }

    if save_output:
        sim_utils.makeFigures(
            quad.params,
            t_all,
            pos_all,
            vel_all,
            quat_all,
            omega_all,
            euler_all,
            w_cmd_all,
            wmotor_all,
            thr_all,
            tor_all,
            sDes_traj_all,
            sDes_calc_all,
        )

        try:
            from utils.animation import sameAxisAnimation

            ifsave = 0
            if start_pos_3d is not None and goal_pos_3d is not None:
                sameAxisAnimation(
                    t_all,
                    traj.wps,
                    pos_all,
                    quat_all,
                    sDes_traj_all,
                    ts,
                    quad.params,
                    traj.xyzType,
                    traj.yawType,
                    ifsave,
                    start_pos=start_pos_3d,
                    goal_pos=goal_pos_3d,
                    sphere_radius=GOAL_THRESHOLD,
                )
            else:
                sameAxisAnimation(
                    t_all,
                    traj.wps,
                    pos_all,
                    quat_all,
                    sDes_traj_all,
                    ts,
                    quad.params,
                    traj.xyzType,
                    traj.yawType,
                    ifsave,
                )
        except Exception as exc:
            print(f"Warning: Could not create animation: {exc}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate quadcopter from recorded CSV trajectory",
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help=(
            "CSV with columns step, linear_velocity, angular_velocity "
            "(normalized actions in [-1, 1])"
        ),
    )
    parser.add_argument(
        "--goals-json",
        type=str,
        default=None,
        help="Map goals JSON; enables green/red spheres in 3D",
    )
    parser.add_argument(
        "--goal-seed",
        type=int,
        default=0,
        help="Seed for choosing one goal from goal_pose_list",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    print(f"\nLoading trajectory from: {csv_path}")
    print("=" * 60)

    simulate_trajectory_from_csv(
        str(csv_path),
        save_output=True,
        goals_json=args.goals_json,
        goal_seed=args.goal_seed,
    )
    plt.show()
    return 0


if __name__ == "__main__":
    if sim_config.orient not in ("NED", "ENU"):
        raise ValueError(
            f"{sim_config.orient} is not a valid orientation. Verify config.py file."
        )
    sys.exit(main())
