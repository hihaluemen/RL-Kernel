# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Regression guard for issue #191.

On a GPU host the compiled extension ``rl_engine._C`` MUST be present and
launchable; a missing or arch-mismatched build must fail loudly here instead of
silently degrading to the pure-PyTorch fallbacks. On a CPU host (no CUDA) the
test skips, so the CPU CI job - which legitimately has no ``_C`` - stays green.
"""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_compiled_extension_present_and_launches():
    # Import directly from the package, NOT from rl_engine.kernels.ops.base, which
    # deliberately swallows the ImportError and falls back to _C = None.
    try:
        from rl_engine import _C
    except ImportError as exc:  # Bug A: the extension was never built
        pytest.fail(
            "rl_engine._C is missing on a GPU host - the CUDA extension was not "
            "built. Install with `pip install --no-build-isolation -e .`. "
            f"Underlying error: {exc}"
        )

    logits = torch.randn(4, 32, device="cuda", dtype=torch.float32)
    token_ids = torch.randint(0, 32, (4,), device="cuda", dtype=torch.long)
    out = _C.fused_logp(logits, token_ids)
    torch.cuda.synchronize()  # Bug B: an arch mismatch surfaces on synchronize
    assert tuple(out.shape) == (4,)
