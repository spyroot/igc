"""Policy heads for the goal-conditioned tool-use agent.

Houses the pointer / candidate-scoring policy that replaces the fixed-width Q-head,
so the action-space width scales with the legal candidates rather than the global
catalog.

Author:
Mus mbayramo@stanford.edu
"""

# Author: Mus mbayramo@stanford.edu
