"""Tool-teaching: LLM-taught few-shot tool acquisition (``docs/ARCHITECTURE.md`` §12).

The teacher (the DeepSeek-V4-Flash endpoint M3 reuses) induces a
:class:`~igc.core.tool_card.ToolCard` for an unknown tool from a few real
``(call -> result/error)`` interactions; this package grounds the card against
real evidence (:mod:`igc.modules.teach.grounding`) and gates safe cold-start
probing (:mod:`igc.modules.teach.safe_probe`) before any card may influence the
pointer policy. Pure stdlib in the offline slice — no torch/network/DeepSeek.

Author:
Mus mbayramo@stanford.edu
"""

# Author: Mus mbayramo@stanford.edu
