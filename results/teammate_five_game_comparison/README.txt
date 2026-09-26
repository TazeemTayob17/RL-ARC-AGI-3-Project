ARC-AGI-3 V0 vs V1 matched five-game comparison

Games:
- sp80
- r11l
- cd82
- ls20
- tn36

Seed:
5209 for both agents

Configured action budget:
200 actions per game

Recorded actions:
201 per game / 1005 total per agent
(the harness loop records one additional boundary action)

Scorer:
ARC-AGI-3 SDK scorecard (sdk_score / scorecard.score)

Results:
V0 (dqn_v0): 0.023517978620019437 SDK score, 1 level completed
V1 (dqn_v1): approximately 0.4904 SDK score, 3 levels completed

Original Slurm comparison job:
60185
