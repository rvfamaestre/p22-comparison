# NO RL vs PPO vs SAC Experiment Report

Base output: `output/method_compare_random`

## Training Summary

| Method | Episodes | Best Episode | Best Speed Variance | Final Speed Variance | Last10 Mean | Last10 Std | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo | 25 | 9 | 1.471419 | 1.877463 | 1.958466 | 0.137710 | nan |
| sac | 25 | 3 | 1.530381 | 1.761804 | 1.853481 | 0.195498 | nan |

## Evaluation Summary

| Method | Tag | HR | Mean Spd | Mean Spd L100 | Spd Var | Spd Var L100 | RMS Acc | RMS Jerk | Min Gap | Collisions | Clamps | Amp. Ratio | Safety |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_rl | h25 | 0.25 | 4.867621 | 5.174902 | 0.630268 | 0.009837 | 0.578083 | 8.682344 | 0.510000 | 0 | 3 | n/a | False |
| no_rl | h50 | 0.50 | 3.744462 | 4.715679 | 1.938621 | 1.141794 | 0.436595 | 5.169320 | 0.510000 | 0 | 3 | n/a | False |
| no_rl | h75 | 0.75 | 2.984148 | 2.887007 | 3.792930 | 3.717681 | 0.420392 | 3.450944 | 0.307407 | 42 | 4 | n/a | False |
| ppo | h25 | 0.25 | 4.703192 | 4.973604 | 0.570667 | 0.018469 | 0.679304 | 10.334037 | 0.510000 | 0 | 3 | n/a | False |
| ppo | h50 | 0.50 | 3.716447 | 4.716852 | 1.830847 | 0.993440 | 0.450910 | 5.530378 | 0.510000 | 0 | 3 | n/a | False |
| ppo | h75 | 0.75 | 2.979402 | 2.879625 | 3.722245 | 3.666269 | 0.420733 | 3.531509 | 0.345337 | 46 | 3 | n/a | False |
| sac | h25 | 0.25 | 4.833861 | 5.163949 | 0.635622 | 0.006326 | 0.635071 | 9.632094 | 0.510000 | 0 | 3 | n/a | False |
| sac | h50 | 0.50 | 3.745371 | 4.719367 | 1.932221 | 0.967708 | 0.454354 | 5.577224 | 0.510000 | 0 | 3 | n/a | False |
| sac | h75 | 0.75 | 2.975242 | 2.865689 | 3.830024 | 3.777145 | 0.423507 | 3.530667 | 0.300000 | 39 | 5 | n/a | False |

## Best Method By Metric And Human Ratio

### h25

- Mean Speed: `no_rl`
- Mean Speed (Last 100): `no_rl`
- Speed Variance: `ppo`
- Speed Var (Last 100): `sac`
- RMS Acc: `no_rl`
- RMS Jerk: `no_rl`
- Min Gap: `no_rl`
- Collision Count: `no_rl`
- Collision Clamp Count: `no_rl`
- Amplification Ratio: `n/a`
- Min Gap Safe: `no_rl`
- Safety Satisfied: `no_rl`

### h50

- Mean Speed: `sac`
- Mean Speed (Last 100): `sac`
- Speed Variance: `ppo`
- Speed Var (Last 100): `sac`
- RMS Acc: `no_rl`
- RMS Jerk: `no_rl`
- Min Gap: `no_rl`
- Collision Count: `no_rl`
- Collision Clamp Count: `no_rl`
- Amplification Ratio: `n/a`
- Min Gap Safe: `no_rl`
- Safety Satisfied: `no_rl`

### h75

- Mean Speed: `no_rl`
- Mean Speed (Last 100): `no_rl`
- Speed Variance: `ppo`
- Speed Var (Last 100): `ppo`
- RMS Acc: `no_rl`
- RMS Jerk: `no_rl`
- Min Gap: `ppo`
- Collision Count: `sac`
- Collision Clamp Count: `ppo`
- Amplification Ratio: `n/a`
- Min Gap Safe: `no_rl`
- Safety Satisfied: `no_rl`
