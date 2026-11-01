"""Reserve-share sweep for the day-ahead optimisation baseline (DAO).

The DAO baseline of the main study fixes the reserve share at rho = 0.3 and
optimises only the hourly V2G set-points. To test whether the comparison with
PPO is sensitive to that fixed choice, the same optimiser is re-run for a grid
of reserve shares on the held-out test days and the stress days. The per-day
best share ("oracle") is also reported: it is an upper bound on what a DAO with
a freely tuned reserve share could reach, because the choice uses the realised
test-day outcome.

Usage: python scripts/dao_rho_sweep.py [rho ...]   (writes results/eval/dao_rho/rho_<x>.json)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v2g.env import V2GTwoAreaEnv
from v2g.controllers import DayAheadOpt
from v2g.evaluate import rollout, daily_metrics
from v2g.profiles import sample_scenario

RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
N_TEST, N_STRESS = 30, 20
rt, rs = np.random.default_rng(2026), np.random.default_rng(777)
SETS = dict(test=[sample_scenario(rt, "test") for _ in range(N_TEST)],
            stress=[sample_scenario(rs, "stress") for _ in range(N_STRESS)])
OUT = "results/eval/dao_rho"; os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    todo = [float(a) for a in sys.argv[1:]] or RHOS      # optional: run a subset in parallel processes
    for rho in todo:
        f = f"{OUT}/rho_{rho:.1f}.json"
        if os.path.exists(f):
            continue
        out = {}
        for split, scs in SETS.items():
            env = V2GTwoAreaEnv(cfg=dict(reg_energy_diag=False), scenario_list=scs)
            M = []
            for k in range(len(scs)):
                rows = rollout(env, DayAheadOpt(rho=rho, maxiter=20))
                m = daily_metrics(rows, env); m.pop("hourly", None)
                M.append(m)
            out[split] = M
        json.dump(out, open(f, "w"))
        print(f"rho={rho:.1f}", {s: round(-10 * float(np.mean([m["reward"] for m in out[s]])), 2) for s in out}, flush=True)
