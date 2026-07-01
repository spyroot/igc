"""Training configuration: the M1 profile registry + adapter specs.

One source of truth for named training profiles (``docs/TRAINING_OPTIMIZATION_PLAN.md``)
so a large-model run cannot silently inherit GPT-2 / small-GPU defaults. Pure stdlib —
importable and offline-testable without torch/peft.

Author:
Mus mbayramo@stanford.edu
"""

# Author: Mus mbayramo@stanford.edu
