# Phase 3 handoff for the group

**Status: Phase 3 experiments and decisions complete. This branch contains Phase 3 only.**

## What I completed

1. Integrated our teammate's selected DQN alongside the original repo DQN. His architecture/exploration research and pretrained checkpoints are preserved and credited in `BASELINE_EXPERIMENT_HISTORY.txt` and `PATCH_MANIFEST.txt`.
2. Ran the selected DQN, current PPO, their learning-disabled controls, and two random agents on 17 development games with three seeds: **306 runs**.
3. Ran **15 additional original-DQN runs** for a longer five-game comparison; reused 15 matching selected-DQN records.
4. Checked our teammate's archived five-game scorecards and saved the raw evidence.
5. Saved scores, action traces, configurations, learning diagnostics and summaries; recorded the group's final decisions.

An earlier 18-run, one-game pilot is retained separately. It is preliminary evidence and is not counted as another full baseline study.

## Agreed setup

- Starting baselines: **teammate's selected DQN and the existing PPO**.
- Accepted split: 17 development games and 8 games reserved from further tuning, in `configs/splits.json`.
- Budget: **1500 scorer-counted actions per run**; learner seeds 0/1/2; environment seed 0.
- Official evidence here: **local ARC SDK scorecards**, not unseen Kaggle/private results. All 25 public games influenced earlier development.
- The user reported that the group and tutors approved this setup. No independent tutor contact is claimed.

### Names that can cause confusion

| Code name | Meaning in this project |
|---|---|
| Original DQN / code V0 | The DQN already in main before the teammate patch; reference comparison only |
| Selected DQN / code V1 | The teammate's DQN, now our assignment starting baseline |
| PPO / code V0 | Our other assignment starting baseline |

The selected DQN's code label V1 does not mean we have already completed the assignment's first improvement stage.

## Main results

Each row covers 51 runs: 17 games × 3 seeds. Scores are mean local game scores; levels are totals across those runs.

| Agent | Mean score | Total levels |
|---|---:|---:|
| Selected DQN | 0.337304 | 9 |
| Selected DQN, learning off | 0.339643 | 9 |
| PPO | 0.066544 | 12 |
| PPO, learning off | 0.065673 | 11 |
| Type-uniform random | 0.021906 | 10 |
| Flat-uniform random | 0.001460 | 5 |

**What this means:** the agents performed training updates, but those updates did not clearly improve scores within this budget. Learning-versus-control uncertainty intervals cross zero. DQN found no progress in 42/51 runs; PPO in 39/51. Mean repeated-transition fractions were 0.55 and 0.443 respectively. Finding useful actions is the main diagnostic target. A higher score and more completed levels are different measurements.

## How this compares with our teammate's results

His archived test used sp80/r11l/cd82/ls20/tn36, seed 5209, and a configured budget of 200. The harness actually counted 201 actions per game. The scorecards confirm original DQN score 0.02352 / one level and selected DQN score 0.49039 / three levels.

Under the supplied defaults, original DQN waits 1000 actions before training, so it cannot train in that short test. Actual gradient counters and resolved configs are absent from his archive. The result supports short-budget whole-system performance, not a claim that online learning alone caused the gain.

Our longer comparison used ft09/g50t/ls20/ar25/sp80, seeds 0/1/2 and 1500 actions, giving both agents time to train. Original DQN mean score was 0.656670 versus selected DQN 0.006773. All differences came from sp80; both scored zero on the other four games. The protocols differ, so neither comparison proves a general winner. **The group still chose the selected DQN as its starting baseline.**

## Decisions for the next phase

The group delegated improvement selection. The first experiment selected for both agents is a small training reward for reaching less-visited observed screens, motivated by sparse rewards/repeated states and [count-based exploration research](https://arxiv.org/abs/1611.04717). This is a hypothesis to test, not a proven gain. The selected DQN already uses action-choice counts; adding a novelty reward to learning is a distinct change.

Phase 4 implements and evaluates that change separately. Phase 4 code, reward-wrapper tests, pilot results and reports are excluded from this Phase 3 branch.

## Where to find the evidence

- [Main results](PHASE3_RESULTS.md)
- [Longer DQN comparison](DQN_V0_V1_COMPARISON.md)
- [Teammate scorecard verification](TEAMMATE_COMPARISON_VERIFICATION.md)
- [Integration review](PHASE3_INTEGRATION_REVIEW.md)
- Raw results: `results/phase3_selected_baseline`, `results/phase3_v0_v1_comparison`, `results/teammate_five_game_comparison`. Each new run has `record.json` and `actions.npy`.

## Limits to preserve in the report

These are short development runs, not convergence or unseen-game evidence. The selected DQN uses fold-specific public checkpoints and currently rejects unknown games until a deployment checkpoint is supplied. The older full-suite supervised distant-cue capacity failure remains unresolved; the focused Phase 3 checks passed, but we do not claim the entire suite passes. The group accepted its baseline choice; no matched RL comparison of architecture finalists is claimed.

## Reproduce the full baseline grid

Use Python 3.12 and the pinned dependencies. From the repo root:

```bash
python scripts/run_phase3.py --games ar25,cd82,dc22,ft09,g50t,ka59,lf52,ls20,m0r0,r11l,re86,sc25,sp80,su15,tn36,tr87,vc33 --seeds 0,1,2 --budget 1500 --dqn-version V1 --threads 1 --out results/phase3_selected_baseline --execute
python scripts/summarize_phase3.py
```

Completed valid records are reused. Existing Phase 3 records were collected before the final commit; their original config/source identifiers are preserved, not rewritten to pretend they were collected at the publication commit.
