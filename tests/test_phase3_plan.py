import pytest

from scripts.run_phase3 import plan


def test_phase3_plan_uses_only_development_games_and_matched_conditions():
    with pytest.raises(ValueError, match="development"):
        plan(["bp35"], [0], 1500, 32, ["dqn"])
    configs = plan(["sp80"], [0, 1], 1500, 32, ["dqn", "dqn_no_updates", "ppo"])
    assert len(configs) == 6
    assert [c.updates_disabled for c in configs[:3]] == [False, True, False]
    assert len({(c.game, c.seed, c.budget, c.arch["id"]) for c in configs}) == 2


def test_selected_dqn_baseline_and_control_match():
    configs = plan(["sp80"], [0], 1500, 32, ["dqn", "dqn_no_updates", "ppo"], "V1")
    assert [c.version for c in configs] == ["V1", "V1", "V0"]
    assert configs[0].resolved().params == configs[1].resolved().params
    assert not configs[0].updates_disabled and configs[1].updates_disabled


def test_summary_refuses_missing_experiment_records():
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from scripts.summarize_phase3 import summarize
    with TemporaryDirectory() as directory:
        with pytest.raises(RuntimeError, match="Incomplete grid"):
            summarize(Path(directory))


def test_dqn_comparison_refuses_missing_records():
    import scripts.check_dqn_comparison as comparison
    from pathlib import Path
    from tempfile import TemporaryDirectory
    previous = comparison.OUT
    try:
        with TemporaryDirectory() as directory:
            comparison.OUT = Path(directory)
            with pytest.raises(FileNotFoundError, match="Comparison incomplete"):
                comparison.summarize()
    finally:
        comparison.OUT = previous
