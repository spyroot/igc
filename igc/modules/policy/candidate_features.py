"""Typed graph/candidate feature encoder for the pointer policy.

This is the small neural bridge from the M1/M2 State contract to M6 candidate
keys. Text embeddings from :class:`ActionCodec` remain supported; these typed
features are the incremental graph upgrade: resource type, parent/relation,
method, URI path hashes, action-target/OEM/collection flags, allowed-method
mask, and local state summary.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateFeatureEncoder(nn.Module):
    """Encode fixed candidate feature tensors into ``[B, N, H]`` embeddings."""

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
        in_dim = part * 7 + numeric_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return candidate embeddings from collated State feature tensors."""
        resource_type = self.type_emb(_bucket(features["candidate_resource_type_id"], self.type_emb))
        parent_type = self.parent_emb(_bucket(features["candidate_parent_type_id"], self.parent_emb))
        relation = self.relation_emb(_bucket(features["candidate_relation_name_id"], self.relation_emb))
        method = self.method_emb(features["candidate_method_id"].clamp(min=0, max=15).long())
        depth = self.depth_emb(features["candidate_depth_bucket"].clamp(min=0, max=15).long())
        path = self.path_emb(_bucket(features["candidate_path_segment_hashes"], self.path_emb)).mean(dim=-2)
        local = self.local_emb(_bucket(features["candidate_local_state_summary"], self.local_emb)).mean(dim=-2)
        numeric = torch.cat([
            features["candidate_has_action_target"].unsqueeze(-1).float(),
            features["candidate_is_collection"].unsqueeze(-1).float(),
            features["candidate_is_oem"].unsqueeze(-1).float(),
            features["candidate_allowed_method_mask"].float(),
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


def merge_text_and_graph_candidates(text_h: torch.Tensor, graph_h: torch.Tensor) -> torch.Tensor:
    """Combine text and graph candidate embeddings in the shared pointer space."""
    if text_h.shape != graph_h.shape:
        raise ValueError(f"text_h {tuple(text_h.shape)} and graph_h {tuple(graph_h.shape)} differ")
    return F.normalize(text_h + graph_h, dim=-1)


def _bucket(values: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
    return values.long().remainder(emb.num_embeddings)

