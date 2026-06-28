# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the Triton batch-invariant selected-logprob kernel (issue #148).

These tests validate that the Triton kernel produces results that:
1. Match the PyTorch reference implementation (correctness).
2. Are bitwise identical across different batch sizes / positions (batch-invariance).
3. Support backward pass (gradient correctness).

All tests are skipped when Triton or CUDA is unavailable (e.g. on Windows or CPU-only).
"""

import pytest
import torch

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton batch-invariant logp requires CUDA device and Triton.",
)

_V = 300


def _reference_logp(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Canonical reference: log_softmax(fp32) + gather."""
    logits_2d = logits.reshape(-1, logits.size(-1)).float()
    target_1d = target_ids.reshape(-1).long()
    log_probs = torch.log_softmax(logits_2d, dim=-1)
    selected = torch.gather(log_probs, dim=-1, index=target_1d.unsqueeze(1)).squeeze(1)
    return selected.reshape(target_ids.shape)


def _make_row(seed: int, vocab: int = _V, device: str = "cuda") -> torch.Tensor:
    """Generate a single deterministic logit row from a seed."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(1, vocab, generator=gen, device=device)


# ---------------------------------------------------------------------------
# 1. Correctness: Triton vs reference
# ---------------------------------------------------------------------------


@requires_triton_cuda
class TestTritonCorrectness:
    """Triton kernel output must match log_softmax + gather reference."""

    def _get_op(self):
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )
        return TritonBatchInvariantLogpOp()

    def test_matches_reference_fp32(self):
        op = self._get_op()
        logits = torch.randn(8, _V, device="cuda")
        target = torch.randint(0, _V, (8,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-5)

    def test_matches_reference_bf16(self):
        op = self._get_op()
        logits = torch.randn(8, _V, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, _V, (8,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits.float(), target)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-4)

    def test_matches_reference_fp16(self):
        op = self._get_op()
        logits = torch.randn(8, _V, device="cuda", dtype=torch.float16)
        target = torch.randint(0, _V, (8,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits.float(), target)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-4)

    def test_large_vocab(self):
        op = self._get_op()
        logits = torch.randn(4, 128256, device="cuda")
        target = torch.randint(0, 128256, (4,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_single_token(self):
        op = self._get_op()
        logits = torch.randn(1, _V, device="cuda")
        target = torch.randint(0, _V, (1,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_3d_logits(self):
        op = self._get_op()
        logits = torch.randn(2, 3, _V, device="cuda")
        target = torch.randint(0, _V, (2, 3), device="cuda")
        out = op(logits, target)
        assert out.shape == (2, 3)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_matches_pytorch_op(self):
        """Triton and PyTorch ops should agree within tolerance."""
        from rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp import (
            NativeBatchInvariantLogpOp,
        )
        triton_op = self._get_op()
        pytorch_op = NativeBatchInvariantLogpOp()
        logits = torch.randn(16, _V, device="cuda")
        target = torch.randint(0, _V, (16,), device="cuda")
        out_triton = triton_op(logits, target)
        out_pytorch = pytorch_op(logits, target)
        assert torch.allclose(out_triton, out_pytorch, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Batch-invariance on GPU via Triton
# ---------------------------------------------------------------------------


@requires_triton_cuda
class TestTritonBatchInvariance:
    """Triton kernel must be bitwise batch-invariant."""

    def _get_op(self):
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )
        return TritonBatchInvariantLogpOp()

    def test_batch_size_1_vs_n(self):
        op = self._get_op()
        row = _make_row(42)
        target = torch.tensor([7], device="cuda")
        result_alone = op(row, target).item()

        for batch_size in [2, 4, 8, 16, 32, 64, 128]:
            batch_logits = torch.randn(batch_size, _V, device="cuda")
            batch_target = torch.randint(0, _V, (batch_size,), device="cuda")
            batch_logits[0] = row.squeeze(0)
            batch_target[0] = target.squeeze(0)
            result_in_batch = op(batch_logits, batch_target)[0].item()
            assert result_alone == result_in_batch, (
                f"Triton drift at batch_size={batch_size}: "
                f"alone={result_alone}, in_batch={result_in_batch}"
            )

    def test_different_positions(self):
        op = self._get_op()
        row = _make_row(99)
        target = torch.tensor([13], device="cuda")
        batch_size = 16
        results = []
        for pos in range(batch_size):
            batch_logits = torch.randn(batch_size, _V, device="cuda")
            batch_target = torch.randint(0, _V, (batch_size,), device="cuda")
            batch_logits[pos] = row.squeeze(0)
            batch_target[pos] = target.squeeze(0)
            results.append(op(batch_logits, batch_target)[pos].item())
        assert all(r == results[0] for r in results), (
            f"Triton position drift: unique = {set(results)}"
        )

    def test_repeated_runs(self):
        op = self._get_op()
        logits = torch.randn(16, _V, device="cuda")
        target = torch.randint(0, _V, (16,), device="cuda")
        results = [op(logits, target) for _ in range(50)]
        for i, r in enumerate(results[1:], 1):
            assert torch.equal(r, results[0]), f"Triton run {i} differs from run 0"

    def test_mixed_batch_content(self):
        op = self._get_op()
        row = _make_row(77)
        target = torch.tensor([25], device="cuda")
        batch_size = 8
        results = []
        for trial_seed in range(20):
            torch.manual_seed(trial_seed * 1000)
            batch_logits = torch.randn(batch_size, _V, device="cuda")
            batch_target = torch.randint(0, _V, (batch_size,), device="cuda")
            batch_logits[3] = row.squeeze(0)
            batch_target[3] = target.squeeze(0)
            results.append(op(batch_logits, batch_target)[3].item())
        assert all(r == results[0] for r in results), (
            f"Triton mixed-batch drift: unique = {set(results)}"
        )


# ---------------------------------------------------------------------------
# 3. Backward / gradient
# ---------------------------------------------------------------------------


@requires_triton_cuda
class TestTritonBackward:
    """Gradient through the Triton op must match reference."""

    def _get_op(self):
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )
        return TritonBatchInvariantLogpOp()

    def test_backward_matches_reference(self):
        op = self._get_op()
        logits = torch.randn(4, _V, device="cuda", requires_grad=True)
        target = torch.randint(0, _V, (4,), device="cuda")

        out = op(logits, target).sum()
        out.backward()
        grad = logits.grad.detach().clone()

        ref_logits = logits.detach().clone().requires_grad_(True)
        ref = _reference_logp(ref_logits, target).sum()
        ref.backward()

        assert torch.allclose(grad, ref_logits.grad, atol=1e-5)

    def test_gradient_batch_invariance(self):
        op = self._get_op()
        row = _make_row(42)
        target = torch.tensor([7], device="cuda")

        logits_alone = row.clone().requires_grad_(True)
        op(logits_alone, target).sum().backward()
        grad_alone = logits_alone.grad.detach().clone()

        for batch_size in [4, 16, 64]:
            batch_logits = torch.randn(batch_size, _V, device="cuda")
            batch_logits[0] = row.squeeze(0)
            batch_logits.requires_grad_(True)
            batch_target = torch.randint(0, _V, (batch_size,), device="cuda")
            batch_target[0] = target.squeeze(0)
            op(batch_logits, batch_target).sum().backward()
            grad_in_batch = batch_logits.grad[0:1].detach().clone()
            assert torch.allclose(grad_alone, grad_in_batch, atol=1e-5), (
                f"Triton gradient drift at batch_size={batch_size}"
            )

    def test_ignored_row_grad_is_zero(self):
        """Ignored rows must have zero gradient across the entire vocab."""
        op = self._get_op()
        logits = torch.randn(4, _V, device="cuda", requires_grad=True)
        target = torch.tensor([0, -100, 2, -100], device="cuda")
        op(logits, target).sum().backward()
        assert torch.equal(logits.grad[1], torch.zeros(_V, device="cuda"))
        assert torch.equal(logits.grad[3], torch.zeros(_V, device="cuda"))

    def test_backward_bf16_input(self):
        """Backward with bf16 logits should match fp32 reference within tolerance."""
        op = self._get_op()
        logits = torch.randn(8, _V, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, _V, (8,), device="cuda")

        op(logits, target).sum().backward()
        grad = logits.grad.detach().clone()

        ref_logits = logits.detach().float().requires_grad_(True)
        _reference_logp(ref_logits, target).sum().backward()

        assert torch.allclose(grad.float(), ref_logits.grad, atol=1e-2)

    def test_backward_fp16_input(self):
        """Backward with fp16 logits should match fp32 reference within tolerance."""
        op = self._get_op()
        logits = torch.randn(8, _V, device="cuda", dtype=torch.float16, requires_grad=True)
        target = torch.randint(0, _V, (8,), device="cuda")

        op(logits, target).sum().backward()
        grad = logits.grad.detach().clone()

        ref_logits = logits.detach().float().requires_grad_(True)
        _reference_logp(ref_logits, target).sum().backward()

        assert torch.allclose(grad.float(), ref_logits.grad, atol=1e-2)


# ---------------------------------------------------------------------------
# 4. ignore_index handling
# ---------------------------------------------------------------------------


@requires_triton_cuda
class TestTritonIgnoreIndex:

    def _get_op(self):
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )
        return TritonBatchInvariantLogpOp()

    def test_ignore_outputs_zero(self):
        op = self._get_op()
        logits = torch.randn(4, _V, device="cuda")
        target = torch.tensor([0, -100, 2, -100], device="cuda")
        out = op(logits, target)
        assert out[1].item() == 0.0
        assert out[3].item() == 0.0
        ref = _reference_logp(logits[[0, 2]], target[[0, 2]])
        assert torch.allclose(out[[0, 2]], ref, atol=1e-5)

    def test_all_ignore(self):
        op = self._get_op()
        logits = torch.randn(4, _V, device="cuda")
        target = torch.full((4,), -100, device="cuda")
        out = op(logits, target)
        assert torch.equal(out, torch.zeros(4, device="cuda"))


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------


@requires_triton_cuda
class TestTritonValidation:

    def _get_op(self):
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )
        return TritonBatchInvariantLogpOp()

    def test_rejects_cpu_tensor(self):
        op = self._get_op()
        with pytest.raises(RuntimeError, match="requires a GPU"):
            op(torch.randn(4, _V), torch.randint(0, _V, (4,)))

    def test_rejects_1d_logits(self):
        op = self._get_op()
        with pytest.raises(ValueError, match="at least 2-D"):
            op(torch.randn(10, device="cuda"), torch.tensor([0], device="cuda"))

    def test_rejects_invalid_target(self):
        op = self._get_op()
        logits = torch.randn(4, _V, device="cuda")
        target = torch.tensor([0, -1, 2, 3], device="cuda")
        with pytest.raises(ValueError, match="outside"):
            op(logits, target)
