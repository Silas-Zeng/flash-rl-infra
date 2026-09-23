from flashrl.ablation import BASELINE, FLASH, run_ablation


def test_algorithm_ablation_is_portable_without_torch():
    baseline = run_ablation(BASELINE, samples=6, seed=11)
    flash = run_ablation(FLASH, samples=6, seed=11)
    assert baseline["tokens"] == flash["tokens"]
    assert flash["proposed_tokens"] > 0
    assert flash["max_inflight_observed"] <= 6
