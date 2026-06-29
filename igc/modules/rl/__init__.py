"""Pure, offline-testable RL helpers for the goal-conditioned learner.

Kept free of env / LLM / CUDA imports so the Q-learning target and HER relabeling
math is unit-tested on CPU without standing up the full agent.

Author:
Mus mbayramo@stanford.edu
"""
from igc.modules.rl.q_targets import q_learning_target, relabel_future

__all__ = ["q_learning_target", "relabel_future"]


# Author: Mus mbayramo@stanford.edu
