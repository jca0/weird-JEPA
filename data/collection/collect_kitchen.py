"""
Franka Kitchen teleop data collection via iPhone (HEBI Mobile I/O + ARKit).

CONTROLS
--------
    B1 (hold/release)    : hold to record and control, release to end episode
    B3 (press to toggle) : toggle gripper open/closed
    Ctrl+C               : stop collection

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
import time
from datetime import datetime
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "raw"

# Phone-to-sim scaling
POS_SCALE = 5.0  
ROT_SCALE = 1.0

# Proportional gains: position/rotation error → velocity command
POS_GAIN = 2.0
ROT_GAIN = 1.0


# Environment
def make_env(tasks: list, render: bool = True):
    import gymnasium
    import gymnasium_robotics  # noqa: F401 — registers envs

    env = gymnasium.make(
        "FrankaKitchen-v1",
        tasks_to_complete=tasks,
        render_mode="human" if render else None,
        terminate_on_tasks_completed=False,
        max_episode_steps=100000,  # Effectively unlimited
    )
    return env


# IK: phone 6-DOF pose → joint velocities (for position-based control)
def get_ee_jacobian(model, data, ee_site_id: int) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    return np.vstack([jacp, jacr])  # (6, nv)


def phone_pose_to_joint_vel(
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    gripper_vel: float,
    model,
    data,
    ee_site_id: int,
) -> np.ndarray:
    """
    Convert target EE pose to joint velocities using Jacobian.
    This creates position-based behavior through proportional control.

    Args:
        target_pos: desired end-effector position (3,)
        target_quat: desired end-effector quaternion (4,) scipy format [x,y,z,w]
        gripper_vel: gripper velocity [-1, 1]
        model: MuJoCo model
        data: MuJoCo data
        ee_site_id: end-effector site ID

    Returns:
        joint velocities [-1, 1]^9
    """
    # Get current EE pose
    current_pos = data.site_xpos[ee_site_id].copy()
    current_mat = data.site_xmat[ee_site_id].reshape(3, 3)
    current_rot = Rotation.from_matrix(current_mat)

    # Position error
    pos_error = target_pos - current_pos

    # Orientation error (axis-angle)
    target_rot = Rotation.from_quat(target_quat)  # scipy format [x,y,z,w]
    rot_error = (target_rot * current_rot.inv()).as_rotvec()

    # Desired task-space velocity (proportional control)
    task_vel = np.concatenate([pos_error * POS_GAIN, rot_error * ROT_GAIN])

    # Jacobian for arm joints only
    J_arm = get_ee_jacobian(model, data, ee_site_id)[:, :7]  # (6, 7)

    # Damped least-squares to get joint velocities
    damping = 0.05
    JJT = J_arm @ J_arm.T
    joint_vel = J_arm.T @ np.linalg.solve(JJT + damping**2 * np.eye(6), task_vel)

    gripper_cmd = np.full(2, gripper_vel)

    return np.clip(np.concatenate([joint_vel, gripper_cmd]), -1.0, 1.0)


# iPhone teleop stream (HEBI SDK + ARKit)
class IOSTeleopStream:
    """
    Reads 6-DOF ARKit pose from an iPhone running the HEBI Mobile I/O app.

    Mirrors LeRobot's IOSPhone class but standalone (no lerobot dependency).
    B1 = hold to record episode (release ends it). B3 = toggle gripper.
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
        self._gripper_open = True  # Start with gripper open
        self._prev_b3 = False      # For edge detection

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

    def _get_b3(self, fbk) -> bool:
        try:
            return bool(fbk[0].io.b.get_int(3))
        except Exception:
            return False

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
        b3 = self._get_b3(fbk)

        # B3 toggle gripper on rising edge
        if b3 and not self._prev_b3:
            self._gripper_open = not self._gripper_open
        self._prev_b3 = b3

        # Rising edge: re-anchor position reference
        if b1 and not self._enabled:
            self._calib_pos = raw_pos.copy()

        if b1:
            pos_cal = self._calib_rot_inv.apply(raw_pos - self._calib_pos)
            rot_cal = self._calib_rot_inv * raw_rot
        else:
            pos_cal = np.zeros(3)
            rot_cal = Rotation.identity()

        # Gripper velocity: -1 to close, +1 to open
        gripper_vel = 1.0 if self._gripper_open else -1.0

        self._enabled = b1
        return {
            "pos":         pos_cal,
            "rotvec":      rot_cal.as_rotvec(),
            "gripper_vel": gripper_vel,
            "enabled":     b1,
        }


# Episode collection
def collect_episode(env, teleop: IOSTeleopStream) -> dict | None:
    """
    Collect one episode. Returns None if the user discards it (Ctrl+C).
    Hold B1 to record and control. Release B1 to end. B3 toggles gripper.
    """
    obs_list, action_list, reward_list, done_list = [], [], [], []

    obs, _ = env.reset()
    raw_obs = obs["observation"].astype(np.float32)

    robot_env  = env.unwrapped.robot_env
    model      = robot_env.model
    data       = robot_env.data
    ee_site_id = model.site("end_effector").id

    print("  [Episode] Hold B1 to start recording and control. Release B1 to end. B3 toggles gripper.")

    step = 0
    started = False
    ref_ee_pos = None
    ref_ee_quat = None

    try:
        while True:
            phone = teleop.get_action()
            if phone is None:
                time.sleep(0.01)
                continue

            if phone["enabled"]:
                # First press: capture EE reference and start recording
                if not started:
                    started = True
                    ref_ee_pos = data.site_xpos[ee_site_id].copy()
                    ref_ee_mat = data.site_xmat[ee_site_id].reshape(3, 3).copy()
                    ref_ee_quat = Rotation.from_matrix(ref_ee_mat).as_quat()
                    print("  [Episode] Recording...")

                # Compute target EE pose from phone delta
                target_pos = ref_ee_pos + phone["pos"] * POS_SCALE
                phone_rot = Rotation.from_rotvec(phone["rotvec"] * ROT_SCALE)
                ref_rot = Rotation.from_quat(ref_ee_quat)
                target_quat = (ref_rot * phone_rot).as_quat()

                action = phone_pose_to_joint_vel(
                    target_pos,
                    target_quat,
                    phone["gripper_vel"],
                    model, data, ee_site_id,
                ).astype(np.float32)
            else:
                # B1 released after recording started → end episode
                if started:
                    break
                time.sleep(0.01)
                continue

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


# HDF5 saving
def save_episode(episode: dict, tasks: list, out_path: Path, episode_idx: int) -> None:
    """Append a single episode to the HDF5 file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if out_path.exists() else "w"
    with h5py.File(out_path, mode) as f:
        if "tasks" not in f.attrs:
            f.attrs["tasks"] = tasks
        f.attrs["n_episodes"] = episode_idx + 1

        grp = f.create_group(f"episode_{episode_idx}")
        for key, arr in episode.items():
            grp.create_dataset(key, data=arr, compression="gzip")

    print(f"  [Episode] Saved → {out_path} (episode {episode_idx})")


# Entry point
def main():
    parser = argparse.ArgumentParser(description="Collect Franka Kitchen demos via iPhone teleop.")
    parser.add_argument(
        "--tasks", nargs="+", default=["microwave"],
        choices=["microwave", "kettle", "bottom_burner", "top_burner",
                 "light_switch", "slide_cabinet", "hinge_cabinet"],
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(args.out) if args.out
        else DATA_DIR / f"kitchen_teleop_{'_'.join(args.tasks)}_{timestamp}.h5"
    )

    print("=" * 60)
    print("Franka Kitchen iPhone Teleop Collection")
    print(f"  tasks:  {args.tasks}")
    print(f"  output: {out_path}")
    print(f"  Recording episodes until Ctrl+C")
    print("=" * 60)

    env    = make_env(args.tasks, render=True)
    teleop = IOSTeleopStream()
    teleop.connect()
    teleop.calibrate()

    ep_idx = 0
    try:
        while True:
            print(f"\n--- Episode {ep_idx + 1} — hold B1 to start ---")
            ep = collect_episode(env, teleop)
            if ep is not None:
                save_episode(ep, args.tasks, out_path, ep_idx)
                ep_idx += 1
    except KeyboardInterrupt:
        print("\n\nCollection stopped by user.")

    env.close()
    print(f"\nTotal episodes collected: {ep_idx}")


if __name__ == "__main__":
    main()
