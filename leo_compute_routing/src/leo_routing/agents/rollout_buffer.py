from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    batch: object
    log_probability: float
    value: float
    reward: float
    next_value: float
    terminal: bool


def compute_gae(rewards, values, next_values, terminals, gamma, gae_lambda):
    rewards, values, next_values = (np.asarray(v, dtype=np.float64) for v in (rewards, values, next_values))
    terminals = np.asarray(terminals, dtype=bool)
    if not (rewards.shape == values.shape == next_values.shape == terminals.shape) or rewards.ndim != 1:
        raise ValueError("GAE arrays must have matching one-dimensional shapes")
    advantages = np.zeros_like(rewards)
    tail = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        continuing = float(not terminals[index])
        delta = rewards[index] + gamma * next_values[index] * continuing - values[index]
        tail = delta + gamma * gae_lambda * continuing * tail
        advantages[index] = tail
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


class RolloutBuffer:
    def __init__(self):
        self.transitions = []

    def append(self, transition):
        if not all(np.isfinite(v) for v in (transition.log_probability, transition.value,
                                             transition.reward, transition.next_value)):
            raise ValueError("Nonfinite rollout transition")
        self.transitions.append(transition)

    def targets(self, settings):
        return compute_gae([t.reward for t in self.transitions], [t.value for t in self.transitions],
                           [t.next_value for t in self.transitions], [t.terminal for t in self.transitions],
                           settings["gamma"], settings["gae_lambda"])

    def __len__(self):
        return len(self.transitions)
