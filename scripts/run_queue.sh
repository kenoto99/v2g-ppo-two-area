#!/bin/bash
# two parallel workers on a 2-core machine
cd "$(dirname "$0")/.."
(
python3 scripts/train_ppo.py --seed 0 --steps 400000 --tag main
python3 scripts/train_ppo.py --seed 1 --steps 400000 --tag main
python3 scripts/train_ppo.py --seed 2 --steps 400000 --tag main
python3 scripts/train_ppo.py --seed 0 --steps 200000 --tag noOV  --cfg '{"w_ov":0.0}'
python3 scripts/train_ppo.py --seed 0 --steps 200000 --tag noF   --cfg '{"w_f":0.0,"w_tie":0.0}'
python3 scripts/train_ppo.py --seed 0 --steps 200000 --tag noDeg --cfg '{"w_deg":0.0}'
) > results/worker_A.log 2>&1 &
(
python3 scripts/train_ppo.py --seed 3 --steps 400000 --tag main
python3 scripts/train_ppo.py --seed 4 --steps 400000 --tag main
python3 scripts/train_ppo.py --seed 1 --steps 200000 --tag noOV  --cfg '{"w_ov":0.0}'
python3 scripts/train_ppo.py --seed 1 --steps 200000 --tag noF   --cfg '{"w_f":0.0,"w_tie":0.0}'
python3 scripts/train_ppo.py --seed 1 --steps 200000 --tag noDeg --cfg '{"w_deg":0.0}'
) > results/worker_B.log 2>&1 &
wait
# reference runs for the ablation (same training length as the ablated variants)
python3 scripts/train_ppo.py --seed 0 --steps 200000 --tag ref200 > results/worker_C.log 2>&1 &
python3 scripts/train_ppo.py --seed 1 --steps 200000 --tag ref200 > results/worker_D.log 2>&1 &
wait
