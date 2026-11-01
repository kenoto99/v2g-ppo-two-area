#!/bin/bash
# Final (v3 environment) training queue. Two workers on a 2-core machine.
# Seeds: main 0-4 (400k steps); ablation and sensitivity 200k steps.
cd "$(dirname "$0")/.."
T=200000
(
for s in 0 2 4; do python3 scripts/train_ppo.py --seed $s --steps 400000 --tag main; done
for s in 0 1 2; do python3 scripts/train_ppo.py --seed $s --steps $T --tag ref200; done
for s in 0 1 2; do python3 scripts/train_ppo.py --seed $s --steps $T --tag noF --cfg '{"w_f":0.0,"w_tie":0.0}'; done
for s in 0 1; do python3 scripts/train_ppo.py --seed $s --steps $T --tag wV25 --cfg '{"w_v":25.0}'; done
for s in 0 1; do python3 scripts/train_ppo.py --seed $s --steps $T --tag wF2 --cfg '{"w_f":2.0}'; done
) > results/worker_A.log 2>&1 &
(
for s in 1 3; do python3 scripts/train_ppo.py --seed $s --steps 400000 --tag main; done
for s in 0 1 2; do python3 scripts/train_ppo.py --seed $s --steps $T --tag squash --squash; done
for s in 0 1 2; do python3 scripts/train_ppo.py --seed $s --steps $T --tag noOV --cfg '{"w_ov":0.0}'; done
for s in 0 1 2; do python3 scripts/train_ppo.py --seed $s --steps $T --tag noDeg --cfg '{"w_deg":0.0}'; done
for s in 0 1; do python3 scripts/train_ppo.py --seed $s --steps $T --tag wV100 --cfg '{"w_v":100.0}'; done
for s in 0 1; do python3 scripts/train_ppo.py --seed $s --steps $T --tag wF8 --cfg '{"w_f":8.0}'; done
) > results/worker_B.log 2>&1 &
wait
