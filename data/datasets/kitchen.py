"""Minari-based dataset for Franka Kitchen (state-only)."""

import numpy as np
import torch
from torch.utils.data import Dataset


JOINT_DIM = 9
JOINT_VEL_DIM = 9
OBJECT_DIM = 21
OBJECT_VEL_DIM = 20
ACTION_DIM = 9


class KitchenStateDataset(Dataset):
    """Sliding-window dataset over Franka Kitchen episodes from Minari.

    Each sample is a contiguous window of (joint_pos, object_pos, action) tuples.
    The observation vector (59,) is split as:
        [0:9]   robot joint positions
        [9:18]  robot joint velocities
        [18:39] object positions/joint states
        [39:59] object velocities
    """

    def __init__(
        self,
        dataset_name="D4RL/kitchen/partial-v2",
        num_steps=4,
        frameskip=1,
        include_velocities=False,
    ):
        super().__init__()
        import minari

        self.num_steps = num_steps
        self.frameskip = frameskip
        self.include_velocities = include_velocities

        ds = minari.load_dataset(dataset_name, download=True)

        self.episodes = []
        self.windows = []

        for ep in ds.iterate_episodes():
            obs = ep.observations["observation"].astype(np.float32)
            actions = ep.actions.astype(np.float32)
            self.episodes.append({"obs": obs, "actions": actions})

            ep_idx = len(self.episodes) - 1
            window_len = (num_steps - 1) * frameskip + 1
            n_windows = len(actions) - window_len + 1
            for start in range(max(n_windows, 0)):
                self.windows.append((ep_idx, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ep_idx, start = self.windows[idx]
        ep = self.episodes[ep_idx]

        indices = [start + i * self.frameskip for i in range(self.num_steps)]

        obs = ep["obs"][indices]  # (num_steps, 59)
        joint_pos = obs[:, :JOINT_DIM]
        object_pos = obs[:, JOINT_DIM + JOINT_VEL_DIM : JOINT_DIM + JOINT_VEL_DIM + OBJECT_DIM]

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
            joint_vel = obs[:, JOINT_DIM : JOINT_DIM + JOINT_VEL_DIM]
            object_vel = obs[:, JOINT_DIM + JOINT_VEL_DIM + OBJECT_DIM :]
            sample["joint_vel"] = torch.from_numpy(joint_vel)
            sample["object_vel"] = torch.from_numpy(object_vel)

        return sample
