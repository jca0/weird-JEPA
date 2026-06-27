"""Dataset wrapper for HDF5 files collected via collect_kitchen.py."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class KitchenTeleopDataset(Dataset):
    """Sliding-window dataset over self-collected Franka Kitchen teleop episodes.

    Each HDF5 file is produced by data/collection/collect_kitchen.py.
    Observation layout matches KitchenStateDataset (59-dim vector).
    """

    def __init__(
        self,
        path: str | Path,
        num_steps: int = 4,
        frameskip: int = 1,
        include_velocities: bool = False,
    ):
        super().__init__()
        import h5py

        self.num_steps = num_steps
        self.frameskip = frameskip
        self.include_velocities = include_velocities

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        JOINT_DIM = 9
        JOINT_VEL_DIM = 9
        OBJECT_DIM = 21

        self.episodes = []
        self.windows = []

        with h5py.File(path, "r") as f:
            n_episodes = f.attrs.get("n_episodes", len(f.keys()))
            for i in range(n_episodes):
                grp = f[f"episode_{i}"]
                obs = grp["obs"][:].astype(np.float32)       # (T, 59)
                actions = grp["action"][:].astype(np.float32) # (T, 9)
                self.episodes.append({"obs": obs, "actions": actions})

                ep_idx = len(self.episodes) - 1
                window_len = (num_steps - 1) * frameskip + 1
                n_windows = len(actions) - window_len + 1
                for start in range(max(n_windows, 0)):
                    self.windows.append((ep_idx, start))

        self._JOINT_DIM = JOINT_DIM
        self._JOINT_VEL_DIM = JOINT_VEL_DIM
        self._OBJECT_DIM = OBJECT_DIM

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ep_idx, start = self.windows[idx]
        ep = self.episodes[ep_idx]

        JD, JVD, OD = self._JOINT_DIM, self._JOINT_VEL_DIM, self._OBJECT_DIM

        indices = [start + i * self.frameskip for i in range(self.num_steps)]
        obs = ep["obs"][indices]  # (num_steps, 59)

        joint_pos  = obs[:, :JD]
        object_pos = obs[:, JD + JVD : JD + JVD + OD]

        action_indices = indices[:-1]
        if self.frameskip > 1:
            actions = np.stack([
                ep["actions"][i : i + self.frameskip].reshape(-1)
                for i in action_indices
            ])
        else:
            actions = ep["actions"][action_indices]

        sample = {
            "joint_pos": torch.from_numpy(joint_pos),
            "object_pos": torch.from_numpy(object_pos),
            "action": torch.from_numpy(actions),
        }

        if self.include_velocities:
            joint_vel  = obs[:, JD : JD + JVD]
            object_vel = obs[:, JD + JVD + OD :]
            sample["joint_vel"] = torch.from_numpy(joint_vel)
            sample["object_vel"] = torch.from_numpy(object_vel)

        return sample
