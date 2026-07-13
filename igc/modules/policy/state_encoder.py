"""Explicit State encoder pieces for M1/M2/M6 integration.

The tensor contract is:

* M1 produces per-resource token hidden states from one resource JSON/text.
* ``ResourceTextPooler`` pools tokens within that one resource.
* ``GraphFeatureEncoder`` embeds typed resource/candidate graph features.
* ``NodeFusion`` fuses JSON and typed-feature embeddings into one node embedding.
* ``StatePooler`` aggregates a mask-selected set of node embeddings into
  ``state_latent``.
* ``CandidateEncoder`` reuses endpoint node embeddings plus candidate features for
  pointer-policy keys.

These modules are intentionally independent of a Hugging Face backbone so the
contract is testable offline.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResourceTextPooler(nn.Module):
    """Mask-aware mean pooler over one resource's token sequence."""

    def forward(self, token_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool ``[B,T,H]`` or ``[B,N,T,H]`` hidden states to ``[B,H]``/``[B,N,H]``."""
        mask = attention_mask.to(token_hidden.dtype).unsqueeze(-1)
        summed = (token_hidden * mask).sum(dim=-2)
        counts = mask.sum(dim=-2).clamp(min=1.0)
        return summed / counts


class GraphFeatureEncoder(nn.Module):
    """Embed typed Redfish resource/candidate graph features to ``[B,N,H]``."""

    def __init__(self, hidden_size: int, feature_buckets: int = 4096, path_buckets: int = 8192):
        super().__init__()
        self.hidden_size = hidden_size
        part = max(8, hidden_size // 8)
        self.type_emb = nn.Embedding(feature_buckets, part)
        self.parent_emb = nn.Embedding(feature_buckets, part)
        self.relation_emb = nn.Embedding(feature_buckets, part)
        self.method_emb = nn.Embedding(16, part)
        self.depth_emb = nn.Embedding(16, part)
        self.path_emb = nn.Embedding(path_buckets, part)
        self.local_emb = nn.Embedding(feature_buckets, part)
        numeric_dim = 3 + 6
        self.proj = nn.Sequential(
            nn.Linear(part * 7 + numeric_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, features: Mapping[str, torch.Tensor], prefix: str = "") -> torch.Tensor:
        """Encode feature tensors.

        ``prefix`` lets the same module consume ``candidate_*`` or ``scope_*`` keys.
        """
        resource_type = self.type_emb(self._bucket(features[f"{prefix}resource_type_id"], self.type_emb))
        parent_type = self.parent_emb(self._bucket(features[f"{prefix}parent_type_id"], self.parent_emb))
        relation = self.relation_emb(self._bucket(features[f"{prefix}relation_name_id"], self.relation_emb))
        method = self.method_emb(features[f"{prefix}method_id"].clamp(min=0, max=15).long())
        depth = self.depth_emb(features[f"{prefix}depth_bucket"].clamp(min=0, max=15).long())
        path = self.path_emb(
            self._bucket(features[f"{prefix}path_segment_hashes"], self.path_emb)
        ).mean(dim=-2)
        local = self.local_emb(
            self._bucket(features[f"{prefix}local_state_summary"], self.local_emb)
        ).mean(dim=-2)
        numeric = torch.cat([
            features[f"{prefix}has_action_target"].unsqueeze(-1).float(),
            features[f"{prefix}is_collection"].unsqueeze(-1).float(),
            features[f"{prefix}is_oem"].unsqueeze(-1).float(),
            features[f"{prefix}allowed_method_mask"].float(),
        ], dim=-1)
        return self.proj(torch.cat([
            resource_type,
            parent_type,
            relation,
            method,
            depth,
            path,
            local,
            numeric,
        ], dim=-1))

    @staticmethod
    def _bucket(values: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
        return values.long().remainder(emb.num_embeddings)


class NodeFusion(nn.Module):
    """Fuse per-resource JSON embedding and typed graph-feature embedding."""

    def __init__(self, json_dim: int, feature_dim: int, node_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(json_dim + feature_dim, node_dim),
            nn.GELU(),
            nn.LayerNorm(node_dim),
        )

    def forward(self, json_emb: torch.Tensor, feat_emb: torch.Tensor) -> torch.Tensor:
        """Return node embeddings with shape matching the leading input dimensions."""
        return self.net(torch.cat([json_emb, feat_emb], dim=-1))


class StatePooler(nn.Module):
    """Mask-aware set pooling over node embeddings in the observation scope."""

    def __init__(self, node_dim: int, state_dim: Optional[int] = None):
        super().__init__()
        state_dim = state_dim or node_dim
        self.net = nn.Sequential(
            nn.Linear(2 * node_dim, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        scope_mask: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool ``[B,N,D]`` node embeddings to ``[B,D_state]`` using ``scope_mask``."""
        del edge_features  # reserved for relation-aware pooling; v1 is explicit masked set pooling.
        mask = scope_mask.to(node_emb.dtype).unsqueeze(-1)
        mean = (node_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        masked = node_emb.masked_fill(mask == 0, float("-inf"))
        maxed = masked.max(dim=1).values
        maxed = torch.where(torch.isfinite(maxed), maxed, torch.zeros_like(maxed))
        return self.net(torch.cat([mean, maxed], dim=-1))


class CandidateEncoder(nn.Module):
    """Build candidate embeddings from action text, endpoint node, and graph features."""

    def __init__(self, text_dim: int, node_dim: int, graph_dim: int, candidate_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + node_dim + graph_dim, candidate_dim),
            nn.GELU(),
            nn.LayerNorm(candidate_dim),
        )

    def forward(
        self,
        action_text_h: torch.Tensor,
        endpoint_node_h: torch.Tensor,
        candidate_graph_h: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[B,N,D]`` candidate embeddings."""
        return F.normalize(
            self.net(torch.cat([action_text_h, endpoint_node_h, candidate_graph_h], dim=-1)),
            dim=-1,
        )

