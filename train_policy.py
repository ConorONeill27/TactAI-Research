"""Collect nominal MuJoCo demonstrations and train a 20 Hz wrench policy."""

import argparse
import hashlib
import json

import numpy as np
import torch

from src.controllers.learned_policy import (
    ACTION_CENTER, ACTION_SCALE, CHECKPOINT, CHUNK, LearnedChunkPolicy, chunk_model,
    damping_wrench, features,
)
from src.envs.franka_slot_env import FrankaSlotEnv

DEMO_SEED = 51031
ID_SEED = 52031
DEMO_COUNT = 96
ID_COUNT = 40
TRAIN_SEED = 51033
EPOCHS = 300


def expert(obs):
    lateral = -1100 * obs["ee_pos"][:2] - 55 * obs["ee_linvel"][:2]
    torque = -12 * obs["ee_rpy"] - .45 * obs["ee_angvel"]
    # Give the attitude loop a short alignment interval before the plunge.
    axial = 24.525 - 32 * obs["ee_linvel"][2]
    if obs["time"] < .10:
        axial -= 200 * obs["ee_pos"][2]
    else:
        axial -= 8
    return np.r_[lateral, axial, torque]


def rollouts(env, seed, count, action, collect=False):
    rng = np.random.default_rng(seed)
    rows, data = [], []
    for i in range(count):
        tilt = rng.uniform(-1.5, 1.5, 2)
        offset = rng.uniform(-.25, .25, 2)
        obs, info = env.reset(seed=seed + i, ood_tilt_deg=tilt, ood_offset_mm=offset)
        states, commands = [], []
        for _ in range(550):
            states.append(features(obs))
            wrench = action(obs)
            commands.append(wrench - damping_wrench(obs))
            obs, _, term, trunc, info = env.step(wrench)
            if term or trunc:
                break
        rows.append(dict(seed=seed + i, tilt_deg=tilt.tolist(), offset_mm=offset.tolist(),
                         success=bool(info["is_success"]), jammed=bool(info["is_jammed"]),
                         peak_force_n=float(info["peak_force_n"]), steps=len(commands)))
        if collect and info["is_success"]:
            for start in range(0, len(commands), CHUNK):
                chunk = np.array(commands[start:start + CHUNK], dtype=np.float32)
                mask = np.arange(CHUNK) < len(chunk)
                chunk = np.pad(chunk, ((0, CHUNK - len(chunk)), (0, 0)), mode="edge")
                data.append((states[start], (chunk - ACTION_CENTER) / ACTION_SCALE, mask))
    return rows, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=type(CHECKPOINT), default=CHECKPOINT)
    args = parser.parse_args()
    if (args.output.parent / "freeze_manifest.json").exists():
        parser.error("Use a separate output directory to train after the experiment is frozen.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(TRAIN_SEED)
    env = FrankaSlotEnv()
    demos, examples = rollouts(env, DEMO_SEED, DEMO_COUNT, expert, collect=True)
    x, y, mask = [torch.from_numpy(np.stack(values)) for values in zip(*examples)]
    model = chunk_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(EPOCHS):
        for batch in torch.randperm(len(x)).split(64):
            diff = model(x[batch]).reshape(-1, CHUNK, 6) - y[batch]
            loss = diff[mask[batch]].square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    policy = LearnedChunkPolicy(model=model.eval())
    validation, _ = rollouts(env, ID_SEED, ID_COUNT, policy.get_action)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    report = dict(demonstrations=demos, nominal_validation=validation, training_seed=TRAIN_SEED,
                  examples=len(examples), epochs=EPOCHS, batch_size=64, learning_rate=.002,
                  final_batch_mse=float(loss.detach()), horizon=CHUNK,
                  checkpoint_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_name("training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Expert successes:", sum(row["success"] for row in demos), "/", len(demos))
    print("Nominal learned successes:", sum(row["success"] for row in validation), "/", len(validation))


if __name__ == "__main__":
    main()
