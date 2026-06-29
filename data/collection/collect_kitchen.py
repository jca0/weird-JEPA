"""
Franka Kitchen teleop data collection via iPhone (HEBI Mobile I/O + ARKit).

CONTROLS
--------
    B1 (hold/release)    : hold to record and control, release to end episode
    B3 (press to toggle) : toggle gripper open/closed
    Ctrl+C               : stop collection

USAGE
-----
    python data/collection/collect_kitchen.py --tasks microwave
    python data/collection/collect_kitchen.py --tasks microwave kettle

OUTPUT
------
    data/raw/kitchen_teleop_<tasks>_<timestamp>.h5

    HDF5 layout:
        /episode_N/obs     float32 (T, 59)  — observation at each step
        /episode_N/action  float32 (T, 9)   — joint position command (normalized)
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
POS_SCALE = 3.0
ROT_SCALE = 3.0

# Low-pass filter smoothing (0 = no smoothing, 1 = frozen)
SMOOTH_ALPHA = 0.7

# IK parameters
IK_ITERS = 50
IK_STEP = 0.5
IK_DAMPING = 0.01
IK_TOL = 1e-3


# Environment
def make_env(tasks: list, render: bool = True):
    import gymnasium
    import gymnasium_robotics  # noqa: F401 — registers envs

    env = gymnasium.make(
        "FrankaKitchen-v1",
        tasks_to_complete=tasks,
        render_mode="human" if render else None,
        terminate_on_tasks_completed=False,
        max_episode_steps=100000,
    )
    return env


# IK: target EE pose → joint positions (radians)
def solve_ik(
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    model,
    data,
    ee_site_id: int,
    seed_qpos: np.ndarray = None,
) -> np.ndarray:
    """
    Solve IK for a target EE pose. Returns 7 joint positions in radians.

    Uses iterative Jacobian IK on a temporary data object so we don't
    corrupt the simulation state. If seed_qpos is provided, starts from
    that configuration for consistency between frames.
    """
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos[:]

    if seed_qpos is not None:
        d.qpos[:7] = seed_qpos

    qpos_arm = d.qpos[:7].copy()
    target_rot = Rotation.from_quat(target_quat)  # scipy [x,y,z,w]

    for _ in range(IK_ITERS):
        d.qpos[:7] = qpos_arm
        mujoco.mj_fwdPosition(model, d)

        cur_pos = d.site_xpos[ee_site_id]
        cur_rot = Rotation.from_matrix(d.site_xmat[ee_site_id].reshape(3, 3))

        pos_err = target_pos - cur_pos
        rot_err = (target_rot * cur_rot.inv()).as_rotvec()
        err = np.concatenate([pos_err, rot_err])

        if np.linalg.norm(err) < IK_TOL:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d, jacp, jacr, ee_site_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])

        JJT = J @ J.T + IK_DAMPING * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, err)
        qpos_arm += IK_STEP * dq

        # Clip to actual joint limits
        for i in range(7):
            qpos_arm[i] = np.clip(qpos_arm[i], model.jnt_range[i, 0], model.jnt_range[i, 1])

    return qpos_arm


# iPhone teleop stream (HEBI SDK + ARKit)
class IOSTeleopStream:
    """
    Reads 6-DOF ARKit pose from an iPhone running the HEBI Mobile I/O app.
    B1 = hold to record episode (release ends it). B3 = toggle gripper.
    """

    CAMERA_OFFSET = np.array([0.0, -0.02, 0.04])

    def __init__(self):
        import hebi
        self._hebi = hebi
        self._group = None
        self._calib_pos: np.ndarray | None = None
        self._calib_rot_inv: Rotation | None = None
        self._enabled = False
        self._gripper_open = True
        self._prev_b3 = False

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
        ar_pos = getattr(pose, "ar_position", None)
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
            if self._get_b1(fbk):
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
        """Returns dict with keys: pos, rotvec, gripper_open, enabled."""
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

        self._enabled = b1
        return {
            "pos":          pos_cal,
            "rotvec":       rot_cal.as_rotvec(),
            "gripper_open": self._gripper_open,
            "enabled":      b1,
        }


# Episode collection
def collect_episode(env, teleop: IOSTeleopStream) -> dict | None:
    """
    Collect one episode. Returns None if the user discards it (Ctrl+C).
    Hold B1 to record and control. Release B1 to end. B3 toggles gripper.

    Bypasses env.step() for control (writes directly to data.ctrl) because
    the env's action normalization clips joint 3's range.
    """
    obs_list, action_list, reward_list, done_list = [], [], [], []

    obs, _ = env.reset()

    robot_env = env.unwrapped.robot_env
    model = robot_env.model
    data = robot_env.data
    ee_site_id = model.site("end_effector").id
    frame_skip = robot_env.frame_skip

    def get_obs():
        robot_obs = robot_env._get_obs()
        return env.unwrapped._get_obs(robot_obs)["observation"].astype(np.float32)

    raw_obs = get_obs()

    print("  [Episode] Hold B1 to start recording and control. Release B1 to end. B3 toggles gripper.")

    step = 0
    started = False
    ref_ee_pos = None
    ref_ee_quat = None
    prev_qpos_target = None
    smooth_pos = np.zeros(3)
    smooth_rotvec = np.zeros(3)

    try:
        while True:
            phone = teleop.get_action()
            if phone is None:
                time.sleep(0.01)
                continue

            if phone["enabled"]:
                if not started:
                    started = True
                    ref_ee_pos = data.site_xpos[ee_site_id].copy()
                    ref_ee_quat = Rotation.from_matrix(
                        data.site_xmat[ee_site_id].reshape(3, 3)
                    ).as_quat()
                    smooth_pos = phone["pos"].copy()
                    smooth_rotvec = phone["rotvec"].copy()
                    print("  [Episode] Recording...")

                # Low-pass filter
                smooth_pos = SMOOTH_ALPHA * smooth_pos + (1 - SMOOTH_ALPHA) * phone["pos"]
                smooth_rotvec = SMOOTH_ALPHA * smooth_rotvec + (1 - SMOOTH_ALPHA) * phone["rotvec"]

                # Direct mapping: phone XYZ → world XYZ
                target_pos = ref_ee_pos + smooth_pos * POS_SCALE
                phone_rot = Rotation.from_rotvec(smooth_rotvec * ROT_SCALE)
                target_quat = (Rotation.from_quat(ref_ee_quat) * phone_rot).as_quat()

                # Solve IK → joint positions (radians)
                qpos_target = solve_ik(target_pos, target_quat, model, data, ee_site_id, prev_qpos_target)
                prev_qpos_target = qpos_target.copy()



                # Gripper target
                gripper_target = 0.04 if phone["gripper_open"] else 0.0

                # Write directly to ctrl (bypasses env's broken normalization)
                data.ctrl[:7] = qpos_target
                data.ctrl[7] = gripper_target
                data.ctrl[8] = gripper_target
            else:
                if started:
                    break
                time.sleep(0.01)
                continue

            # Step the simulation
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)

            if robot_env.render_mode == "human":
                robot_env.render()

            # Record
            action = np.concatenate([qpos_target, [gripper_target, gripper_target]])
            new_obs = get_obs()

            obs_list.append(raw_obs)
            action_list.append(action.astype(np.float32))
            reward_list.append(0.0)
            done_list.append(False)

            raw_obs = new_obs
            step += 1

            time.sleep(0.02)  # ~50 Hz

    except KeyboardInterrupt:
        print("  [Episode] Discarded.")
        return None

    print(f"  [Episode] Complete — {step} steps")
    return {
        "obs":    np.array(obs_list, dtype=np.float32),
        "action": np.array(action_list, dtype=np.float32),
        "reward": np.array(reward_list, dtype=np.float32),
        "done":   np.array(done_list, dtype=bool),
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
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer window.")
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

    env = make_env(args.tasks, render=args.render)
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
        print("\n\nCollection stopped.")

    env.close()
    print(f"\nTotal episodes collected: {ep_idx}")


if __name__ == "__main__":
    main()
