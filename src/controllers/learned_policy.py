"""Proprioceptive behavior cloning with 50-tick wrench chunks."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

CHUNK = 50
CHECKPOINT = Path(__file__).resolve().parents[2] / "data" / "learned_policy" / "weights.pt"
FEATURE_SCALE = np.array([.00025, .00025, .025, .026, .026, .026,
                          .02, .02, .4, 1, 1, 1, .15], dtype=np.float32)
ACTION_CENTER = np.array([0, 0, 20, 0, 0, 0], dtype=np.float32)
ACTION_SCALE = np.array([1, 1, 4, .12, .12, .12], dtype=np.float32)
DAMPING = np.array([55, 55, 32, .45, .45, .45])


def damping_wrench(obs):
    return -DAMPING * np.concatenate([obs["ee_linvel"], obs["ee_angvel"]])


def features(obs):
    return np.concatenate([obs["ee_pos"], obs["ee_rpy"], obs["ee_linvel"],
                           obs["ee_angvel"], [obs["time"]]]).astype(np.float32) / FEATURE_SCALE


def chunk_model():
    return nn.Sequential(nn.Linear(13, 64), nn.Tanh(), nn.Linear(64, 64),
                         nn.Tanh(), nn.Linear(64, CHUNK * 6))


@lru_cache(maxsize=2)
def load_model(path):
    model = chunk_model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


class LearnedChunkPolicy:
    def __init__(self, path=CHECKPOINT, model=None):
        self.model = model if model is not None else load_model(Path(path).resolve())
        self.reset()

    def reset(self, seed=None):
        self.chunk = None

    def get_action(self, obs):
        index = int(obs["step"]) % CHUNK
        if index == 0 or self.chunk is None:
            with torch.inference_mode():
                output = self.model(torch.from_numpy(features(obs))).numpy().reshape(CHUNK, 6)
            self.chunk = output * ACTION_SCALE + ACTION_CENTER
        return self.chunk[index].astype(np.float64) + damping_wrench(obs)
