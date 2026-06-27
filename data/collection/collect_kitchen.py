"""
Franka Kitchen teleop data collection via iPhone (HEBI Mobile I/O + ARKit).

OVERVIEW
--------
Run this script on your LOCAL MACHINE — not EC2. The iPhone and your laptop
must be on the same WiFi. The iPhone streams 6-DOF ARKit pose over UDP via the
HEBI Mobile I/O app; this script converts pose deltas to Franka joint velocities
using Jacobian IK, steps the MuJoCo sim, and saves episodes to HDF5.

After collecting, copy data to EC2 for training:
    scp data/raw/kitchen_teleop_*.h5 ubuntu@<EC2_IP>:~/workspace/my-JEPA/data/raw/

SETUP
-----
1. Clone this repo locally and create a Python 3.10 venv:
       python3.10 -m venv .venv && source .venv/bin/activate
       pip install hebi-py gymnasium gymnasium-robotics mujoco h5py scipy

2. Install the free HEBI Mobile I/O app on your iPhone (App Store).
   In the app settings: Family = "HEBI", Name = "mobileIO".

3. Connect your iPhone and laptop to the same WiFi network.

CONTROLS
--------
    B1 (press & hold)  : enable teleoperation. On the first press, the current
                         phone pose is captured as the reference (zero position).
                         Releasing B1 pauses the robot mid-episode.
    A3 (analog slider) : gripper — slide up to close, down to open, center to hold.
    Ctrl+C             : discard the current episode and retry.

USAGE
-----
    python data/collection/collect_kitchen.py --tasks microwave --episodes 20
    python data/collection/collect_kitchen.py --tasks microwave kettle --episodes 30

OUTPUT
------
    data/raw/kitchen_teleop_<tasks>_<timestamp>.h5

    HDF5 layout:
        /episode_N/obs     float32 (T, 59)  — observation at each step
        /episode_N/action  float32 (T, 9)   — joint velocity command sent
        /episode_N/reward  float32 (T,)     — task reward
        /episode_N/done    bool    (T,)     — episode termination flag

    Observation vector layout (matches KitchenStateDataset):
        [ 0: 9]  robot joint positions
        [ 9:18]  robot joint velocities
        [18:39]  object positions / joint states
        [39:59]  object velocities
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "raw"

# Scale phone position deltas (meters) → joint velocity magnitude
POS_GAIN     = 3.0
ROT_GAIN     = 1.5
GRIPPER_GAIN = 1.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def make_env(tasks: list, render: bool = False):
    import gymnasium
    import gymnasium_robotics  # noqa: F401 — registers envs

    env = gymnasium.make(
        "FrankaKitchen-v1",
        tasks_to_complete=tasks,
        render_mode="human" if render else None,
        terminate_on_tasks_completed=False,
    )
    return env


# ---------------------------------------------------------------------------
# IK: phone 6-DOF delta → 9D joint velocity
# ---------------------------------------------------------------------------

def get_ee_jacobian(model, data, ee_site_id: int) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    return np.vstack([jacp, jacr])  # (6, nv)


def phone_pose_to_joint_vel(
    pos_delta: np.ndarray,
    rotvec_delta: np.ndarray,
    gripper_vel: float,
    model,
    data,
    ee_site_id: int,
) -> np.ndarray:
    """Damped least-squares Jacobian IK. Returns clipped action in [-1, 1]^9."""
    J_arm = get_ee_jacobian(model, data, ee_site_id)[:, :7]  # (6, 7)
    task_vel = np.concatenate([pos_delta * POS_GAIN, rotvec_delta * ROT_GAIN])
    damping = 0.05
    JJT = J_arm @ J_arm.T
    dls = J_arm.T @ np.linalg.solve(JJT + damping ** 2 * np.eye(6), task_vel)
    gripper_cmd = np.full(2, gripper_vel * GRIPPER_GAIN)
    return np.clip(np.concatenate([dls, gripper_cmd]), -1.0, 1.0)


# ---------------------------------------------------------------------------
# iPhone teleop stream (HEBI SDK + ARKit)
# ---------------------------------------------------------------------------

class IOSTeleopStream:
    """
    Reads 6-DOF ARKit pose from an iPhone running the HEBI Mobile I/O app.

    Mirrors LeRobot's IOSPhone class but standalone (no lerobot dependency).
    B1 = enable/disable. A3 = gripper analog.
    """

    # iPhone 14 Pro camera is offset from physical center
    CAMERA_OFFSET = np.array([0.0, -0.02, 0.04])

    def __init__(self):
        import hebi
        self._hebi = hebi
        self._group = None
        self._calib_pos: np.ndarray | None = None
        self._calib_rot_inv: Rotation | None = None
        self._enabled = False

    def connect(self):
        print("Searching for HEBI Mobile I/O... (make sure the app is open and on the same WiFi)")
        lookup = self._hebi.Lookup()
        time.sleep(2.0)
        group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
        if group is None:
            raise RuntimeError(
                "HEBI Mobile I/O not found.\n"
                "  - Is the app open on your iPhone?\n"
                "  - Family='HEBI', Name='mobileIO' set in the app?\n"
                "  - iPhone and laptop on the same WiFi?"
            )
        self._group = group
        print(f"Connected to HEBI group ({group.size} module).")

    def calibrate(self):
        """Block until B1 is pressed; captures reference pose."""
        print("\n[Calibration] Hold the phone with:")
        print("  - Top edge pointing FORWARD (same direction as robot +x)")
        print("  - Screen facing UP (robot +z)")
        print("Then press and HOLD B1 in the HEBI app...\n")
        pos, rot = self._wait_for_b1()
        self._calib_pos = pos.copy()
        self._calib_rot_inv = rot.inv()
        self._enabled = False
        print("[Calibration] Done.\n")

    def _read_pose(self):
        fbk = self._group.get_next_feedback()
        if fbk is None:
            return False, None, None, None
        pose = fbk[0]
        ar_pos  = getattr(pose, "ar_position",    None)
        ar_quat = getattr(pose, "ar_orientation", None)
        if ar_pos is None or ar_quat is None:
            return False, None, None, fbk
        # HEBI gives w,x,y,z; scipy wants x,y,z,w
        quat_xyzw = np.concatenate((ar_quat[1:], [ar_quat[0]]))
        rot = Rotation.from_quat(quat_xyzw)
        pos = ar_pos - rot.apply(self.CAMERA_OFFSET)
        return True, pos, rot, fbk

    def _wait_for_b1(self):
        while True:
            ok, pos, rot, fbk = self._read_pose()
            if not ok:
                time.sleep(0.01)
                continue
            b1 = self._get_b1(fbk)
            if b1:
                return pos, rot
            time.sleep(0.01)

    def _get_b1(self, fbk) -> bool:
        try:
            return bool(fbk[0].io.b.get_int(1))
        except Exception:
            return False

    def _get_a3(self, fbk) -> float:
        try:
            return float(fbk[0].io.a.get_float(3))
        except Exception:
            return 0.0

    @property
    def is_calibrated(self) -> bool:
        return self._calib_pos is not None

    def get_action(self):
        """
        Returns dict with keys: pos, rotvec, gripper_vel, enabled.
        Returns None if no valid pose available yet.
        """
        ok, raw_pos, raw_rot, fbk = self._read_pose()
        if not ok or not self.is_calibrated:
            return None

        b1 = self._get_b1(fbk)
        gripper_vel = self._get_a3(fbk)

        # Rising edge: re-anchor position reference
        if b1 and not self._enabled:
            self._calib_pos = raw_pos.copy()

        if b1:
            pos_cal = self._calib_rot_inv.apply(raw_pos - self._calib_pos)
            rot_cal = self._calib_rot_inv * raw_rot
        else:
            pos_cal = np.zeros(3)
            rot_cal = Rotation.identity()

        self._enabled = b1
        return {
            "pos":         pos_cal,
            "rotvec":      rot_cal.as_rotvec(),
            "gripper_vel": gripper_vel,
            "enabled":     b1,
        }


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def collect_episode(env, teleop: IOSTeleopStream, max_steps: int = 500) -> dict | None:
    """
    Collect one episode. Returns None if the user discards it (Ctrl+C).
    Hold B1 to drive the robot; release to pause. A3 controls gripper.
    """
    obs_list, action_list, reward_list, done_list = [], [], [], []

    obs, _ = env.reset()
    raw_obs = obs["observation"].astype(np.float32)

    robot_env  = env.unwrapped.robot_env
    model      = robot_env.model
    data       = robot_env.data
    ee_site_id = model.site("end_effector").id

    print("  [Episode] Hold B1 and move phone to control. Release B1 to pause. Ctrl+C to discard.")
    step = 0
    try:
        while step < max_steps:
            phone = teleop.get_action()

            if phone is None or not phone["enabled"]:
                action = np.zeros(9, dtype=np.float32)
            else:
                action = phone_pose_to_joint_vel(
                    phone["pos"],
                    phone["rotvec"],
                    phone["gripper_vel"],
                    model, data, ee_site_id,
                ).astype(np.float32)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            obs_list.append(raw_obs)
            action_list.append(action)
            reward_list.append(float(reward))
            done_list.append(bool(terminated or truncated))

            raw_obs = next_obs["observation"].astype(np.float32)
            step += 1

            if terminated or truncated:
                break

            time.sleep(0.02)  # ~50 Hz

    except KeyboardInterrupt:
        print("  [Episode] Discarded.")
        return None

    print(f"  [Episode] Complete — {step} steps, total reward={sum(reward_list):.2f}")
    return {
        "obs":    np.array(obs_list,    dtype=np.float32),
        "action": np.array(action_list, dtype=np.float32),
        "reward": np.array(reward_list, dtype=np.float32),
        "done":   np.array(done_list,   dtype=bool),
    }


# ---------------------------------------------------------------------------
# HDF5 saving
# ---------------------------------------------------------------------------

def save_episodes(episodes: list, tasks: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["tasks"]      = tasks
        f.attrs["n_episodes"] = len(episodes)
        for i, ep in enumerate(episodes):
            grp = f.create_group(f"episode_{i}")
            for key, arr in ep.items():
                grp.create_dataset(key, data=arr, compression="gzip")
    print(f"\nSaved {len(episodes)} episodes → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect Franka Kitchen demos via iPhone teleop.")
    parser.add_argument(
        "--tasks", nargs="+", default=["microwave"],
        choices=["microwave", "kettle", "bottom_burner", "top_burner",
                 "light_switch", "slide_cabinet", "hinge_cabinet"],
    )
    parser.add_argument("--episodes",  type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--render",    action="store_true", help="Show MuJoCo GUI")
    parser.add_argument("--out",       type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(args.out) if args.out
        else DATA_DIR / f"kitchen_teleop_{'_'.join(args.tasks)}_{timestamp}.h5"
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 60)
    print("Franka Kitchen iPhone Teleop Collection")
    print(f"  tasks:    {args.tasks}")
    print(f"  episodes: {args.episodes}")
    print(f"  output:   {out_path}")
    print("=" * 60)

    env    = make_env(args.tasks, render=args.render)
    teleop = IOSTeleopStream()
    teleop.connect()
    teleop.calibrate()

    episodes = []
    ep_idx   = 0
    while ep_idx < args.episodes:
        print(f"\n--- Episode {ep_idx + 1}/{args.episodes} ---")
        ep = collect_episode(env, teleop, max_steps=args.max_steps)
        if ep is not None:
            episodes.append(ep)
            ep_idx += 1
        else:
            ans = input("  Retry? [Y/n]: ").strip().lower()
            if ans == "n":
                ep_idx += 1

    env.close()

    if episodes:
        save_episodes(episodes, args.tasks, out_path)
    else:
        print("No episodes collected.")


if __name__ == "__main__":
    main()
