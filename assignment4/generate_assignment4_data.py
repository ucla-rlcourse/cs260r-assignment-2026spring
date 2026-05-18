#!/usr/bin/env python3
"""
WARNING: DO NOT MODIFY THIS FILE!
This files shows how assignment4_data.pt is generated. You do not need to read this file to complete the assignment.

Generate danger-aware preference data for MetaDrive Assignment 4.
Usage: python generate_assignment4_data.py
Saves: assignment4_data.pt in the same directory.

Data collection strategy:
  - A biased agent (expert + steering bias + noise) drives the car.
  - When the car approaches road boundaries (danger zone), we record
    preference pairs: (state, expert_action+noise, biased_action).
  - After danger, the expert takes over briefly to recover the car,
    then the biased agent resumes.

Saved data:
  - Demonstration data (biased agent rollouts): for BC training
  - Recovery data (expert actions during recovery): for HG-DAgger training
  - Preference pairs (danger-aware pos/neg actions): for DPO training
"""
import sys, os

import numpy as np
import torch
from collections import deque
from tqdm import tqdm


# ===== Dataset Classes =====
class DemoDataset:
    """Demonstration dataset for Behavior Cloning training.

    Attributes:
        states:  torch.Tensor [N, state_dim] - observation states
        actions: torch.Tensor [N, action_dim] - demonstration actions from a biased agent
    """
    def __init__(self, states: torch.Tensor, actions: torch.Tensor):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    @classmethod
    def load(cls, path='assignment4_data.pt'):
        data = torch.load(path)
        return cls(data['demo_states'], data['demo_actions'])


class RecoveryDataset:
    """Recovery dataset from expert interventions (HG-DAgger).

    Contains states and expert actions recorded when the expert took over
    to correct dangerous situations during data collection.

    Attributes:
        states:  torch.Tensor [N, state_dim] - states during expert recovery
        actions: torch.Tensor [N, action_dim] - expert corrective actions
    """
    def __init__(self, states: torch.Tensor, actions: torch.Tensor):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    @classmethod
    def load(cls, path='assignment4_data.pt'):
        data = torch.load(path)
        return cls(data['recov_states'], data['recov_actions'])


class PreferenceDataset:
    """Preference dataset for DPO training.

    Each entry is a preference pair (state, pos_action, neg_action) where
    pos_action is preferred over neg_action.

    Attributes:
        states:      torch.Tensor [N, state_dim] - states where danger was detected
        pos_actions: torch.Tensor [N, action_dim] - preferred actions (near-expert, corrective)
        neg_actions: torch.Tensor [N, action_dim] - dispreferred actions (biased, led to danger)
    """
    def __init__(self, states: torch.Tensor, pos_actions: torch.Tensor, neg_actions: torch.Tensor):
        self.states = states
        self.pos_actions = pos_actions
        self.neg_actions = neg_actions

    def __len__(self):
        return len(self.states)

    @classmethod
    def load(cls, path='assignment4_data.pt'):
        data = torch.load(path)
        return cls(data['pref_states'], data['pref_pos'], data['pref_neg'])

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert.numpy_expert import expert
from metadrive.constants import TerminationState

# === Config ===
NUM_SCENARIOS = 100
TOTAL_EPS = 30
K_LOOKBACK = 10
DANGER_TH = 2.0
RECOVERY_STEPS = 15
STEERING_BIAS = 0.2
NOISE_STD = 0.2
RANDOM_PROB = 0.1
POS_NOISE = 0.15

DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(DIR, 'assignment4_data.pt')

# === Setup ===
env = MetaDriveEnv(dict(
    map=3, traffic_density=0.1, num_scenarios=NUM_SCENARIOS,
    start_seed=0, horizon=2000, use_render=False,
))
obs, _ = env.reset(seed=0)
sdim = obs.shape[0]
adim = env.action_space.shape[0]
print(f"State dim: {sdim}, Action dim: {adim}")

# === Collect Data ===
np.random.seed(42)

# Demonstration data (biased agent)
demo_states, demo_actions = [], []
# Recovery data (expert driving)
recov_states, recov_actions = [], []
# Preference pairs
pref_s, pref_p, pref_n = [], [], []

n_danger = 0
rollout_rets, rollout_srs = [], []

for ep in tqdm(range(TOTAL_EPS), desc="Collecting data"):
    obs, _ = env.reset(seed=ep % NUM_SCENARIOS)
    done = False
    ep_r = 0
    buffer = deque(maxlen=K_LOOKBACK)
    recovery_counter = 0

    while not done:
        v = env.vehicle
        dl = v.dist_to_left_side if v.dist_to_left_side is not None else 999
        dr = v.dist_to_right_side if v.dist_to_right_side is not None else 999
        min_dist = min(dl, dr)

        exp_action = expert(v, deterministic=True)

        if recovery_counter > 0:
            # Recovery mode: expert drives to save the car
            action = exp_action
            recovery_counter -= 1
            # Save recovery data
            recov_states.append(obs.copy())
            recov_actions.append(exp_action.copy())
        else:
            # Biased mode
            if np.random.random() < RANDOM_PROB:
                biased = np.random.uniform(-1, 1, adim)
            else:
                biased = exp_action.copy()
                biased[0] += STEERING_BIAS
                biased = np.clip(biased + np.random.normal(0, NOISE_STD, adim), -1, 1)

            # Save demonstration data
            demo_states.append(obs.copy())
            demo_actions.append(biased.copy())

            # Add to lookback buffer
            pos_a = np.clip(exp_action + np.random.normal(0, POS_NOISE, adim), -1, 1)
            buffer.append((obs.copy(), pos_a, biased.copy()))

            # Danger detection
            if min_dist < DANGER_TH:
                n_danger += 1
                for (s, a_pos, a_neg) in buffer:
                    pref_s.append(s)
                    pref_p.append(a_pos)
                    pref_n.append(a_neg)
                pos_now = np.clip(exp_action + np.random.normal(0, POS_NOISE, adim), -1, 1)
                pref_s.append(obs.copy())
                pref_p.append(pos_now)
                pref_n.append(biased.copy())
                buffer.clear()
                recovery_counter = RECOVERY_STEPS
                action = exp_action
            else:
                action = biased

        obs, r, tm, tr, info = env.step(action)
        ep_r += r
        done = tm or tr
        if done:
            rollout_srs.append(float(info.get(TerminationState.SUCCESS, False)))
    rollout_rets.append(ep_r)

env.close()

# === Save ===
data = {
    # Demonstration data (biased agent's states and actions)
    'demo_states': torch.tensor(np.array(demo_states), dtype=torch.float32),
    'demo_actions': torch.tensor(np.array(demo_actions), dtype=torch.float32),
    # Recovery data (expert actions during recovery)
    'recov_states': torch.tensor(np.array(recov_states), dtype=torch.float32),
    'recov_actions': torch.tensor(np.array(recov_actions), dtype=torch.float32),
    # Preference pairs
    'pref_states': torch.tensor(np.array(pref_s), dtype=torch.float32),
    'pref_pos': torch.tensor(np.array(pref_p), dtype=torch.float32),
    'pref_neg': torch.tensor(np.array(pref_n), dtype=torch.float32),
}
torch.save(data, SAVE_PATH)

pe_dist = torch.sqrt(((data['pref_pos'] - data['pref_neg'])**2).sum(-1)).mean().item()
print(f"\n{'='*50}")
print(f"Data Generation Complete")
print(f"{'='*50}")
print(f"Demonstration samples: {len(demo_states)}")
print(f"Preference pairs:      {len(pref_s)}")
print(f"Saved to: {SAVE_PATH}")
