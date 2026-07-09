"""Offline tests for the best-attention model loader in ``llm_shared``.

``_from_pretrained_best_attention`` requests the fastest attention kernel the env
supports (FlashAttention-2 on CUDA, then SDPA) and falls back to the default eager path
when a kernel is unavailable — so it is a safe no-op on CPU / GPT-2 while giving the GPU
profiles FlashAttention on GB300. These tests fake the model class so nothing downloads.

Author:
Mus mbayramo@stanford.edu
"""
from igc.modules.shared import llm_shared


class _FakeModel:
    """Records the attn_implementation it was constructed with."""

    def __init__(self, attn):
        self.attn = attn


def _install_fake(monkeypatch, accept_attn: bool):
    """Install a fake AutoModelForCausalLM; record every from_pretrained call."""
    calls = []

    class _FakeAuto:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            attn = kwargs.get("attn_implementation")
            calls.append(attn)
            if attn is not None and not accept_attn:
                raise ValueError(f"unsupported attn_implementation={attn}")
            return _FakeModel(attn)

    monkeypatch.setattr(llm_shared, "AutoModelForCausalLM", _FakeAuto)
    return calls


def test_falls_back_to_eager_when_kernels_unsupported(monkeypatch):
    """When SDPA/FA2 are rejected, it retries with no attn kwarg (eager) and succeeds."""
    calls = _install_fake(monkeypatch, accept_attn=False)
    model = llm_shared._from_pretrained_best_attention("dummy", {"trust_remote_code": True})
    assert isinstance(model, _FakeModel)
    assert model.attn is None                     # final successful call passed no attn kwarg
    assert "sdpa" in calls                          # it did attempt SDPA first (CPU: no FA2)
    assert calls[-1] is None                        # last attempt is the eager fallback


def test_uses_sdpa_when_supported_on_cpu(monkeypatch):
    """On CPU (no CUDA) the first attempted kernel is SDPA, and it is used when accepted."""
    calls = _install_fake(monkeypatch, accept_attn=True)
    model = llm_shared._from_pretrained_best_attention("dummy", {})
    assert model.attn == "sdpa"
    assert calls == ["sdpa"]                        # accepted on first try, no fallback needed


# Author: Mus mbayramo@stanford.edu
