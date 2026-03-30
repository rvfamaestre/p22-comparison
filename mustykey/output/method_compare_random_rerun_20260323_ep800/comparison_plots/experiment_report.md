# NO RL vs PPO vs SAC Experiment Report

Base output: `output/method_compare_random_rerun_20260323_ep800`

## Training Summary

| Method | Episodes | Best Episode | Best Speed Variance | Final Speed Variance | Last10 Mean | Last10 Std | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo | 800 | 389 | 1.815495 | 1.881215 | 1.881221 | 0.003035 | 28399.750000 |
| sac | 800 | 1 | 1.894264 | 1.940089 | 1.938264 | 0.004104 | 27275.470000 |

## Evaluation Summary

| Method | Tag | Human Ratio | Mean Speed | Speed Variance | RMS Acc | RMS Jerk | Min Gap | Collision Count | Collision Clamp Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_rl | h25 | 0.25 | 4.867621 | 0.630268 | 0.578083 | 8.682344 | 0.510000 | 0 | 3 |
| no_rl | h50 | 0.50 | 3.744462 | 1.938621 | 0.436595 | 5.169320 | 0.510000 | 0 | 3 |
| no_rl | h75 | 0.75 | 2.984148 | 3.792930 | 0.420392 | 3.450944 | 0.307407 | 42 | 4 |
| ppo | h25 | 0.25 | 5.080795 | 0.779726 | 0.441422 | 6.301476 | 0.510000 | 0 | 3 |
| ppo | h50 | 0.50 | 3.727021 | 1.819449 | 0.439357 | 5.256603 | 0.510000 | 0 | 3 |
| ppo | h75 | 0.75 | 2.991476 | 3.607461 | 0.419376 | 3.519632 | 0.489036 | 1 | 3 |
| sac | h25 | 0.25 | 4.822307 | 0.629566 | 0.745568 | 11.362532 | 0.510000 | 0 | 3 |
| sac | h50 | 0.50 | 3.754933 | 1.915758 | 0.476911 | 6.011286 | 0.510000 | 0 | 3 |
| sac | h75 | 0.75 | 2.977267 | 3.828580 | 0.425611 | 3.590832 | 0.300000 | 42 | 5 |

## Best Method By Metric And Human Ratio

### h25

- Mean Speed: `ppo`
- Speed Variance: `sac`
- RMS Acc: `ppo`
- RMS Jerk: `ppo`
- Min Gap: `no_rl`
- Collision Count: `no_rl`
- Collision Clamp Count: `no_rl`

### h50

- Mean Speed: `sac`
- Speed Variance: `ppo`
- RMS Acc: `no_rl`
- RMS Jerk: `no_rl`
- Min Gap: `no_rl`
- Collision Count: `no_rl`
- Collision Clamp Count: `no_rl`

### h75

- Mean Speed: `ppo`
- Speed Variance: `ppo`
- RMS Acc: `ppo`
- RMS Jerk: `no_rl`
- Min Gap: `ppo`
- Collision Count: `ppo`
- Collision Clamp Count: `ppo`
