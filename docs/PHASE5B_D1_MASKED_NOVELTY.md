# D1: was the I1 novelty signal swamped by volatile pixels?

Replayed 102 stored I1 traces (results/phase4_i1, development games, 1,500 actions) with no learner. Replays that did not reproduce the stored score and action count: 0.

Mask rule (online, frames only): after 20 steps, a pixel is volatile if it changed in at least 80% of the steps of every action type tried 3+ times. 'full' is the exact-frame key I1 used; 'masked' ignores volatile pixels (online = mask known at that step, final = mask at the end of the run).

| Game | Runs | Masked pixels (mean) | Unique states full / masked-online / masked-final | New-state rate full -> masked-final | Bonus-factor std full -> masked-final |
|---|---:|---:|---|---|---|
| ar25 | 6 | 0.0 | 721 / 721 / 721 | 0.49 -> 0.49 | 0.250 -> 0.250 |
| cd82 | 6 | 0.0 | 826 / 826 / 826 | 0.56 -> 0.56 | 0.196 -> 0.196 |
| dc22 | 6 | 0.0 | 637 / 637 / 637 | 0.43 -> 0.43 | 0.205 -> 0.205 |
| ft09 | 6 | 0.0 | 120 / 120 / 120 | 0.08 -> 0.08 | 0.230 -> 0.230 |
| g50t | 6 | 0.0 | 652 / 652 / 652 | 0.44 -> 0.44 | 0.222 -> 0.222 |
| ka59 | 6 | 0.0 | 978 / 978 / 978 | 0.66 -> 0.66 | 0.165 -> 0.165 |
| lf52 | 6 | 0.0 | 369 / 369 / 369 | 0.25 -> 0.25 | 0.258 -> 0.258 |
| ls20 | 6 | 0.0 | 980 / 980 / 980 | 0.66 -> 0.66 | 0.179 -> 0.179 |
| m0r0 | 6 | 0.0 | 684 / 684 / 684 | 0.46 -> 0.46 | 0.198 -> 0.198 |
| r11l | 6 | 0.0 | 1262 / 1262 / 1262 | 0.87 -> 0.87 | 0.180 -> 0.180 |
| re86 | 6 | 0.0 | 1272 / 1272 / 1272 | 0.86 -> 0.86 | 0.104 -> 0.104 |
| sc25 | 6 | 0.0 | 670 / 670 / 670 | 0.45 -> 0.45 | 0.242 -> 0.242 |
| sp80 | 6 | 0.0 | 817 / 817 / 817 | 0.56 -> 0.56 | 0.212 -> 0.212 |
| su15 | 6 | 0.0 | 284 / 284 / 284 | 0.19 -> 0.19 | 0.286 -> 0.286 |
| tn36 | 6 | 0.0 | 927 / 927 / 927 | 0.63 -> 0.63 | 0.265 -> 0.265 |
| tr87 | 6 | 0.0 | 1430 / 1430 / 1430 | 0.96 -> 0.96 | 0.070 -> 0.070 |
| vc33 | 6 | 0.0 | 254 / 254 / 254 | 0.17 -> 0.17 | 0.261 -> 0.261 |

Games where masking removes at least 20% of the unique states: none.

No volatile pixels were found in any game, so masking changes nothing: the high unique-state counts are not counters or status bars. A direct check of tr87 (96% new states, 4 actions) shows every action changes about 22 pixels, 244 pixels ever change, and each takes at most two values: its states are genuinely different combinations of a binary pattern. Weak point W6 (state key swamped by counters) is therefore withdrawn.

What remains true: in r11l, re86, tr87 more than 85% of steps land in a new state, so the count bonus 1/sqrt(n) is almost always 1 and cannot tell actions apart (its standard deviation is 0.07 to 0.18 there). That is a property of count-based novelty in huge state spaces, not a state-key bug.
