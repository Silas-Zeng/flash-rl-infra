import pytest

torch = pytest.importorskip("torch")

from flashrl.speculative import (
    run_mtp_smoke,
    speculative_sample,
    verify_candidates,
    verify_speculative_tokens,
)


def test_speculative_accepts_matching_draft_and_emits_bonus_token():
    target = torch.tensor([[0.2, 0.8], [0.3, 0.7]], dtype=torch.float32)
    draft = target.clone()
    candidates = torch.tensor([1, 1], dtype=torch.long)
    trace = verify_speculative_tokens(
        target,
        draft,
        candidates,
        target_next_probs=torch.tensor([0.6, 0.4]),
        generator=torch.Generator().manual_seed(4),
    )

    assert trace.proposed == 2
    assert trace.accepted == 2
    assert trace.rejected_at is None
    assert trace.correction_token is None
    assert trace.bonus_token in (0, 1)
    assert trace.emitted[:2] == (1, 1)
    assert trace.acceptance_probs == (1.0, 1.0)


def test_speculative_sample_draws_candidates_from_draft_before_verification():
    target = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    draft = torch.tensor([[0.5, 0.5], [0.2, 0.8]], dtype=torch.float32)
    candidates, trace = speculative_sample(
        target,
        draft,
        generator=torch.Generator().manual_seed(5),
    )

    assert candidates.shape == (2,)
    assert ((candidates >= 0) & (candidates < 2)).all()
    assert trace.proposed == 2
    assert len(trace.acceptance_probs) <= 2


def test_speculative_rejection_draws_from_residual_distribution():
    # At the first position p(token=0)=0 and q(token=0)=1.  The candidate is
    # therefore rejected with probability one, and max(p-q, 0) is token 1.
    target = torch.tensor([[0.0, 1.0], [0.5, 0.5]], dtype=torch.float32)
    draft = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    candidates = torch.tensor([0, 1], dtype=torch.long)
    trace = verify_speculative_tokens(target, draft, candidates)

    assert trace.accepted == 0
    assert trace.rejected_at == 0
    assert trace.correction_token == 1
    assert trace.emitted == (1,)
    assert trace.acceptance_probs == (0.0,)


def test_speculative_accepts_prefix_then_corrects_first_rejection():
    target = torch.tensor([[0.2, 0.8], [1.0, 0.0]], dtype=torch.float32)
    draft = torch.tensor([[0.2, 0.8], [0.0, 1.0]], dtype=torch.float32)
    candidates = torch.tensor([1, 1], dtype=torch.long)
    trace = verify_speculative_tokens(target, draft, candidates)

    assert trace.accepted == 1
    assert trace.rejected_at == 1
    assert trace.emitted == (1, 0)
    assert trace.correction_token == 0
    assert trace.bonus_token is None


def test_verify_candidates_logprob_wrapper_uses_standard_sampling():
    target = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    draft = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    trace = verify_candidates(
        target.log(),
        torch.tensor([0, 0]),
        draft_logprobs=draft.log(),
    )

    assert trace.rejected_at == 0
    assert trace.correction_token == 1


def test_speculative_rejects_bad_shapes_and_candidate_ids():
    target = torch.tensor([[0.5, 0.5]])
    draft = target.clone()
    with pytest.raises(ValueError, match="same shape"):
        verify_speculative_tokens(target, torch.ones(1, 3), torch.tensor([0]))
    with pytest.raises(ValueError, match="out-of-range"):
        verify_speculative_tokens(target, draft, torch.tensor([2]))


def test_mtp_smoke_performs_one_loss_decreasing_update():
    result = run_mtp_smoke(hidden_size=8, vocab_size=16, future_tokens=2)

    assert result["optimization_steps"] == 1
    assert result["optimized"] is True
    assert result["loss_after"] < result["loss_before"]

